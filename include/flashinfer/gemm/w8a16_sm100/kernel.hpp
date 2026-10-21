/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "flashinfer/gemm/w8a16_sm100/policy.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/detail/cluster.hpp"
#include "cutlass/arch/grid_dependency_control.h"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm100_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<
    cutlass::detail::is_kernel_tag_of_v<typename CollectiveMainloop_::DispatchPolicy::Schedule,
                                KernelW8A16MixedInputTransformSm100>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;

  // Get Blk and Scheduling tile shapes
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;
  using AtomThrShapeMNK = typename CollectiveMainloop::AtomThrShapeMNK;

  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static constexpr bool IsComplex = DispatchPolicy::InputTransformType == cutlass::gemm::detail::KernelInputTransformType::InterleavedComplexTF32;
  static_assert(ArchTag::kMinComputeCapability >= 100);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  // CLC pipeline depth
  // determines how many waves (stages-1) a warp can race ahead
  static constexpr uint32_t SchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  // TileID scheduler
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, CtaShape_MNK, ClusterShape, SchedulerPipelineStageCount>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

  static constexpr bool IsDynamicCluster = not cute::is_static_v<ClusterShape>;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads           = NumThreadsPerWarp;                             // 1 warp
  static constexpr uint32_t NumMMAThreads             = NumThreadsPerWarp;                             // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads    = NumThreadsPerWarp;                             // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads    = NumThreadsPerWarp;                             // 1 warp
  static constexpr uint32_t NumEpilogueThreads        = CollectiveMainloop::NumAccumThreads;           // 4 warps
  static constexpr uint32_t NumEpilogueWarps          = NumEpilogueThreads / NumThreadsPerWarp;
  static constexpr uint32_t NumTransformationThreads  = CollectiveMainloop::NumTransformationThreads;  // 8 warps (two transform warpgroups)
  static constexpr uint32_t NumMainloopLoadBThreads   = NumThreadsPerWarp;                            // 1 warp

  static constexpr uint32_t MaxThreadsPerBlock = NumSchedThreads +
                                                 NumMainloopLoadThreads + NumMMAThreads +
                                                 NumEpilogueLoadThreads +
                                                 NumEpilogueThreads + NumTransformationThreads + NumMainloopLoadBThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  static constexpr uint32_t AccumulatorPipelineStageCount = DispatchPolicy::Schedule::AccumulatorPipelineStageCount;
  static constexpr cutlass::gemm::detail::KernelInputTransformType InputTransformType = DispatchPolicy::InputTransformType;
  static constexpr uint32_t NumFixupBarriers = 1;
  static constexpr uint32_t CLCResponseSize = sizeof(typename TileScheduler::CLCResponse);

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;

  // Pipeline and pipeline state types
  using Load2TransformPipeline = typename CollectiveMainloop::Load2TransformPipeline;
  using Load2TransformPipelineState = typename CollectiveMainloop::Load2TransformPipelineState;

  using Load2MmaPipeline = typename CollectiveMainloop::Load2MmaPipeline;
  using Load2MmaPipelineState = typename CollectiveMainloop::Load2MmaPipelineState;

  using Transform2MmaPipeline = typename CollectiveMainloop::Transform2MmaPipeline;
  using Transform2MmaPipelineState = typename CollectiveMainloop::Transform2MmaPipelineState;

  using Mma2AccumPipeline = typename CollectiveMainloop::Mma2AccumPipeline;
  using Mma2AccumPipelineState = typename CollectiveMainloop::Mma2AccumPipelineState;

  using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
  using EpiLoadPipelineState = typename CollectiveEpilogue::LoadPipelineState;

  using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
  using EpiStorePipelineState = typename CollectiveEpilogue::StorePipelineState;

  using LoadOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  using CLCPipeline = cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape>;
  using CLCPipelineState = cutlass::PipelineState<SchedulerPipelineStageCount>;

  using CLCThrottlePipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount>;
  using CLCThrottlePipelineState = typename CLCThrottlePipeline::PipelineState;

  using TmemAllocator = cute::conditional_t<cute::size(cute::shape<0>(typename TiledMma::ThrLayoutVMNK{})) == 1,
      cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using LoadOrderBarrierStorage = typename LoadOrderBarrier::SharedStorage;
      using CLCPipelineStorage = typename CLCPipeline::SharedStorage;
      using CLCThrottlePipelineStorage = typename CLCThrottlePipeline::SharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) LoadOrderBarrierStorage load_order;
      alignas(16) CLCPipelineStorage clc;
      alignas(16) CLCThrottlePipelineStorage clc_throttle;
      alignas(16) arch::ClusterBarrier tmem_dealloc;
      alignas(16) arch::ClusterBarrier epilogue_throttle;
    } pipelines;

    alignas(16) typename TileScheduler::CLCResponse clc_response[SchedulerPipelineStageCount];
    uint32_t tmem_base_ptr;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;

      EpilogueTensorStorage epilogue;
      MainloopTensorStorage mainloop;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
    KernelHardwareInfo hw_info{};
  };

  enum class WarpCategory : int32_t {
    MMA           = 0,
    Sched         = 1,
    MainloopLoad  = 2,
    EpilogueLoad  = 3,
    Epilogue      = 4,
    // Transformation starts at 256 thread alignment and spans two warpgroups (8 warps)
    Transformation = 8,
    MainloopLoadB  = 16,
  };

  struct IsParticipant {
    uint32_t mma            = false;
    uint32_t sched          = false;
    uint32_t main_load      = false;
    uint32_t main_loadA     = false;
    uint32_t main_loadB     = false;
    uint32_t epi_load       = false;
    uint32_t epilogue       = false;
    uint32_t transformation = false;
  };

  //
  // Methods
  //

  // Convert the host arguments to kernel params: builds the mainloop, epilogue and scheduler
  // params and lays out the workspace they share.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    static constexpr uint32_t NumEpilogueSubTiles = 1;
    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    // Epilogue
    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;

    // Tile scheduler
    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace, args.hw_info),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      TileScheduler::to_underlying_arguments(problem_shape_MNKL, TileShape{}, AtomThrShapeMNK{}, ClusterShape{},
        args.hw_info, args.scheduler, scheduler_workspace
      )
      ,args.hw_info
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);

    if constexpr (IsDynamicCluster) {
      static constexpr int MaxClusterSize = 16;
      implementable &= size(args.hw_info.cluster_shape) <= MaxClusterSize;
      implementable &= size(args.hw_info.cluster_shape_fallback) <= MaxClusterSize;
      implementable &= cutlass::detail::preferred_cluster_can_implement<AtomThrShapeMNK>(args.hw_info.cluster_shape, args.hw_info.cluster_shape_fallback);
    }

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    static constexpr uint32_t NumEpilogueSubTiles = 1;
    size_t workspace_size = 0;

    // Epilogue
    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    // Tile scheduler
    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    static constexpr uint32_t NumEpilogueSubTiles = 1;

    // Epilogue
    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    // Tile scheduler
    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumFixupBarriers, NumEpilogueSubTiles, CollectiveEpilogue::NumAccumulatorMtxs);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, params.hw_info.cluster_shape);
    auto blk_shape = CtaShape_MNK{};
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return TileScheduler::get_grid_shape(
        params.scheduler,
        problem_shape_MNKL,
        TileShape{},
        AtomThrShapeMNK{},
        cluster_shape,
        params.hw_info);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator() (Params const& params, char* smem_buf) {

    using namespace cute;
    using X = Underscore;

    static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes, "SMEM usage exceeded capacity.");
    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Account for multiple epilogue and transformation warps
    int warp_idx = canonical_warp_idx_sync();
    WarpCategory warp_category = warp_category_of(warp_idx);

    int thread_idx          = int(threadIdx.x);
    int thread_idx_in_warp  = thread_idx % 32;
    uint32_t lane_predicate = cute::elect_one_sync();
    int cta_rank_in_cluster = cute::block_rank_in_cluster();
    auto cluster_shape = cutlass::detail::select_cluster_shape(ClusterShape{}, cute::cluster_shape());
    int cluster_size                = size(cluster_shape);
    bool is_first_cta_in_cluster    = (cta_rank_in_cluster == 0);
    bool is_mma_leader_cta          = (cta_rank_in_cluster % size<0>(TiledMma{}) == 0);
    // Even if this variable is unused, shape_div still performs useful compile-time checks.
    [[maybe_unused]] auto mma_leader_ctas = size(shape_div(cluster_shape, AtomThrShapeMNK{}));
    constexpr bool has_mma_peer_cta = cute::size(AtomThrShapeMNK{}) == 2;
    uint32_t mma_peer_cta_rank = has_mma_peer_cta ? cta_rank_in_cluster ^ 1 : cta_rank_in_cluster;

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_category == WarpCategory::Sched) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }
    if ((warp_category == WarpCategory::EpilogueLoad) && lane_predicate) {
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    CollectiveMainloop collective_mainloop(params.mainloop, cluster_shape, cta_rank_in_cluster);
    CollectiveEpilogue collective_epilogue{params.epilogue, shared_storage.tensors.epilogue};

    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
    IsParticipant is_participant;
    is_participant.mma            = (warp_category == WarpCategory::MMA);
    is_participant.sched          = (warp_category == WarpCategory::Sched) && is_first_cta_in_cluster;
    is_participant.main_load      = (warp_category == WarpCategory::MainloopLoad || warp_category == WarpCategory::MainloopLoadB);
    is_participant.main_loadA     = (warp_category == WarpCategory::MainloopLoad);
    is_participant.main_loadB     = (warp_category == WarpCategory::MainloopLoadB);
    is_participant.epi_load       = (warp_category == WarpCategory::EpilogueLoad) && is_epi_load_needed;
    is_participant.epilogue       = (warp_category == WarpCategory::Epilogue);
    is_participant.transformation = (warp_category == WarpCategory::Transformation);

    // MainloopLoad <--> Transformation Pipeline
    Load2TransformPipeline load2transform_pipeline(shared_storage.pipelines.mainloop.load2transform_pipeline,
                                                   load2transform_params(warp_category, thread_idx_in_warp == 0),
                                                   cluster_shape,
                                                   McastDirection::kRow,
                                                   cute::true_type{},  // Perform barrier init
                                                   cute::false_type{}  // Delay mask calculation
                                                   );

    Load2TransformPipelineState load2transform_pipeline_consumer_state;
    Load2TransformPipelineState load2transform_pipeline_producer_state = cutlass::make_producer_start_state<Load2TransformPipeline>();

    // MainloopLoad <--> MMA Pipeline
    Load2MmaPipeline load2mma_pipeline(shared_storage.pipelines.mainloop.load2mma_pipeline,
                                                   load2mma_params(warp_category, lane_predicate && is_mma_leader_cta && is_participant.main_loadB),
                                                   cluster_shape,
                                                   McastDirection::kCol,
                                                   cute::true_type{},  // Perform barrier init
                                                   cute::false_type{}  // Delay mask calculation
                                                   );

    Load2MmaPipelineState load2mma_pipeline_consumer_state;
    Load2MmaPipelineState load2mma_pipeline_producer_state = cutlass::make_producer_start_state<Load2MmaPipeline>();


    // Transformation <--> MMA pipeline
    Transform2MmaPipeline transform2mma_pipeline(shared_storage.pipelines.mainloop.transform2mma_pipeline,
                                                 transform2mma_params(warp_category),
                                                 cluster_shape,
                                                 cute::true_type{},  // Perform barrier init
                                                 cute::false_type{}  // Delay mask calculation
                                                 );

    Transform2MmaPipelineState transform2mma_pipeline_consumer_state;
    Transform2MmaPipelineState transform2mma_pipeline_producer_state = cutlass::make_producer_start_state<Transform2MmaPipeline>();

    // MMA <--> Accumulator pipeline
    Mma2AccumPipeline mma2accum_pipeline(shared_storage.pipelines.mainloop.mma2accum_pipeline,
                                         mma2accum_params(warp_category),
                                         cluster_shape,
                                         cute::true_type{},  // Perform barrier init
                                         cute::false_type{}  // Delay mask calculation
                                         );

    Mma2AccumPipelineState mma2accum_pipeline_consumer_state;
    Mma2AccumPipelineState mma2accum_pipeline_producer_state = cutlass::make_producer_start_state<Mma2AccumPipeline>();

    // Epilogue Load pipeline
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_params(warp_category, cta_rank_in_cluster));

    // Epilogue Store pipeline
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    // Load order barrier
    LoadOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, load_order_params(warp_category));

    EpiLoadPipelineState epi_load_pipe_consumer_state;
    EpiLoadPipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();

    // epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    EpiStorePipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    // CLC pipeline
    // Operates Scheduling Warp <--> All Warps
    CLCPipeline clc_pipeline(shared_storage.pipelines.clc, clc_params(warp_category, cluster_size, is_epi_load_needed), cluster_shape);

    CLCPipelineState clc_pipeline_consumer_state;
    CLCPipelineState clc_pipeline_producer_state = cutlass::make_producer_start_state<CLCPipeline>();

    // CLC throttle pipeline
    CLCThrottlePipeline clc_throttle_pipeline(shared_storage.pipelines.clc_throttle, clc_throttle_params(warp_category));
    CLCThrottlePipelineState clc_pipe_throttle_consumer_state;
    CLCThrottlePipelineState clc_pipe_throttle_producer_state = cutlass::make_producer_start_state<CLCThrottlePipeline>();

    // Tmem allocator
    TmemAllocator tmem_allocator{};

    // Sync allocation status between transform, MMA, and epilogue warps within CTA
    arch::NamedBarrier tmem_allocation_result_barrier(NumTransformationThreads + NumMMAThreads + NumEpilogueThreads,
                                                          cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
    // Sync deallocation status between MMA warps of peer CTAs
    arch::ClusterBarrier& tmem_deallocation_result_barrier = shared_storage.pipelines.tmem_dealloc;
    [[maybe_unused]] uint32_t dealloc_barrier_phase = 0;
    if (WarpCategory::MMA == warp_category && has_mma_peer_cta && lane_predicate) {
      tmem_deallocation_result_barrier.init(NumMMAThreads);
    }

    // Initialize smem barrier for prologue throttling. Epilogue warps are stalled until the prologue finishes.
    arch::ClusterBarrier& epilogue_throttle_barrier = shared_storage.pipelines.epilogue_throttle;
    if (WarpCategory::MMA == warp_category && lane_predicate) {
      epilogue_throttle_barrier.init(                          NumMMAThreads +
                                    (is_first_cta_in_cluster ? NumSchedThreads : 0) +
                                                               NumMainloopLoadThreads +
                                                               NumMainloopLoadBThreads +
                                    (is_epi_load_needed      ? NumEpilogueLoadThreads : 0) +
                                                               NumTransformationThreads);
    }



    // We need this to guarantee that the Pipeline init is visible
    // To all producers and consumer threadblocks in the cluster
    pipeline_init_arrive_relaxed(cluster_size);

    dim3 block_id_in_cluster = cute::block_id_in_cluster();

    // Calculate mask after cluster barrier arrival
    load2transform_pipeline.init_masks(cluster_shape, block_id_in_cluster, cutlass::McastDirection::kRow);
    load2mma_pipeline.init_masks(cluster_shape, cutlass::McastDirection::kCol);
    transform2mma_pipeline.init_masks(cluster_shape);
    mma2accum_pipeline.init_masks(cluster_shape);

    // TileID scheduler
    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    typename TileScheduler::WorkTileInfo work_tile_info = scheduler.initial_work_tile_info(cluster_shape);

    auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);

    // Allocate accumulators
    auto acc_shape = collective_mainloop.partition_accumulator_shape();
    auto bulk_tmem = TiledMma::make_fragment_C(append(acc_shape,
                                                      Int<AccumulatorPipelineStageCount>{}));

    // Tile transform inputs now to get the k tile count
    auto transform_inputs = collective_mainloop.transform_init(params.mainloop, problem_shape_MNKL, bulk_tmem, shared_storage.tensors.mainloop);
    Tensor gA_mkl = get<0>(transform_inputs);

    // Synchronization call. Blocks wait until barriers are initialized in shared memory.
    pipeline_init_wait(cluster_size);

    if (is_participant.main_load) {
      run_main_load(params, shared_storage, is_participant, lane_predicate, is_first_cta_in_cluster,
                    is_epi_load_needed, problem_shape_MNKL, gA_mkl, collective_mainloop, scheduler,
                    work_tile_info,
                    load2transform_pipeline, load2transform_pipeline_producer_state,
                    load2mma_pipeline, load2mma_pipeline_producer_state,
                    clc_pipeline, clc_pipeline_consumer_state,
                    clc_throttle_pipeline, clc_pipe_throttle_producer_state,
                    load_order_barrier, epilogue_throttle_barrier);
    }
    else if (is_participant.sched) {
      run_sched(scheduler, work_tile_info,
                clc_pipeline, clc_pipeline_producer_state, clc_pipeline_consumer_state,
                clc_throttle_pipeline, clc_pipe_throttle_consumer_state,
                epilogue_throttle_barrier);
    }
    else if (is_participant.transformation) {
      run_transform(problem_shape_MNKL, gA_mkl, collective_mainloop, bulk_tmem, transform_inputs,
                    scheduler, work_tile_info,
                    load2transform_pipeline, load2transform_pipeline_consumer_state,
                    transform2mma_pipeline, transform2mma_pipeline_producer_state,
                    clc_pipeline, clc_pipeline_consumer_state,
                    tmem_allocation_result_barrier, epilogue_throttle_barrier);
    }
    else if (is_participant.mma) {
      run_mma(shared_storage, problem_shape_MNKL, collective_mainloop, bulk_tmem, scheduler,
              work_tile_info, is_mma_leader_cta, mma_peer_cta_rank, dealloc_barrier_phase,
              tmem_allocator, tmem_allocation_result_barrier, tmem_deallocation_result_barrier,
              epilogue_throttle_barrier,
              load2mma_pipeline, load2mma_pipeline_consumer_state,
              transform2mma_pipeline, transform2mma_pipeline_consumer_state,
              mma2accum_pipeline, mma2accum_pipeline_producer_state,
              clc_pipeline, clc_pipeline_consumer_state);
    }
    else if (is_participant.epi_load) {
      run_epilogue_load(params, shared_storage, problem_shape_MNKL, collective_epilogue, scheduler,
                        work_tile_info, cta_coord_mnkl,
                        epi_load_pipeline, epi_load_pipe_producer_state,
                        epi_store_pipeline, epi_store_pipe_producer_state,
                        clc_pipeline, clc_pipeline_consumer_state,
                        load_order_barrier, epilogue_throttle_barrier);
    }
    else if (is_participant.epilogue) {
      run_epilogue_store(shared_storage, problem_shape_MNKL, collective_mainloop, collective_epilogue,
                         bulk_tmem, scheduler, work_tile_info, cta_coord_mnkl,
                         epi_load_pipeline, epi_load_pipe_consumer_state,
                         epi_store_pipeline, epi_store_pipe_producer_state,
                         mma2accum_pipeline, mma2accum_pipeline_consumer_state,
                         clc_pipeline, clc_pipeline_consumer_state,
                         tmem_allocation_result_barrier, epilogue_throttle_barrier);
    }
  }

  //
  // The per-role loops of operator().  Each warp runs exactly one; a role receives the
  // pipelines it produces or consumes, its own pipeline states (by value: nothing after the
  // role reads them) and the barriers it arrives on or waits for.  Every role keeps its own
  // fetch-next placement, tail drain and barrier sequence.
  //

  // Which of the role groups a warp belongs to (the enumerators are the first warp of each).
  static CUTLASS_DEVICE WarpCategory
  warp_category_of(int warp_idx) {
    if (warp_idx < static_cast<int>(WarpCategory::Epilogue))       return WarpCategory(warp_idx);
    if (warp_idx < static_cast<int>(WarpCategory::Transformation)) return WarpCategory::Epilogue;
    if (warp_idx < static_cast<int>(WarpCategory::MainloopLoadB))  return WarpCategory::Transformation;
    return WarpCategory::MainloopLoadB;
  }

  // Main-load warps: the A warp feeds the transform pipeline (and the CLC throttle), the B
  // warp the MMA pipeline, each in a prologue of up to Stages k-tiles followed by the rest.
  template <class ProblemShapeMNKL, class TensorA>
  CUTLASS_DEVICE void
  run_main_load(Params const& params, SharedStorage& shared_storage, IsParticipant const& is_participant,
                uint32_t lane_predicate, bool is_first_cta_in_cluster, bool is_epi_load_needed,
                ProblemShapeMNKL const& problem_shape_MNKL, TensorA const& gA_mkl,
                CollectiveMainloop& collective_mainloop, TileScheduler& scheduler,
                typename TileScheduler::WorkTileInfo work_tile_info,
                Load2TransformPipeline& load2transform_pipeline,
                Load2TransformPipelineState load2transform_pipeline_producer_state,
                Load2MmaPipeline& load2mma_pipeline, Load2MmaPipelineState load2mma_pipeline_producer_state,
                CLCPipeline& clc_pipeline, CLCPipelineState clc_pipeline_consumer_state,
                CLCThrottlePipeline& clc_throttle_pipeline,
                CLCThrottlePipelineState clc_pipe_throttle_producer_state,
                LoadOrderBarrier& load_order_barrier, arch::ClusterBarrier& epilogue_throttle_barrier) {
    using namespace cute;

    // Ensure that the prefetched kernel does not touch
    // unflushed global memory prior to this instruction
    cutlass::arch::wait_on_dependent_grids();

    bool do_load_order_arrive = is_epi_load_needed;
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop, shared_storage.tensors.mainloop);

    // Signal the epilogue warps to proceed once the prologue is complete
    epilogue_throttle_barrier.arrive();
    bool requires_clc_query = true;

    do {
      auto cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
      auto k_tile_iter = scheduler.get_k_tile_iterator(work_tile_info, problem_shape_MNKL, CtaShape_MNK{}, shape<3>(gA_mkl));
      auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
      auto k_tile_prologue = min(Load2TransformPipeline::Stages, k_tile_count);

      if(is_participant.main_loadA){
        if constexpr (IsSchedDynamicPersistent) {
          if (is_first_cta_in_cluster && requires_clc_query) {
            clc_throttle_pipeline.producer_acquire(clc_pipe_throttle_producer_state);
            clc_throttle_pipeline.producer_commit(clc_pipe_throttle_producer_state);
            ++clc_pipe_throttle_producer_state;
          }
        }
      }

      if (lane_predicate) {
        if(is_participant.main_loadA){
          auto [load2transform_pipeline_producer_state_next, k_tile_iter_next] = collective_mainloop.load_A(
            params.mainloop,
            load2transform_pipeline,
            load2transform_pipeline_producer_state,
            load_inputs,
            cta_coord_mnkl,
            k_tile_iter, k_tile_prologue
          );
          load2transform_pipeline_producer_state = load2transform_pipeline_producer_state_next;

          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }

          auto [load2transform_pipeline_producer_state_next_, unused_] = collective_mainloop.load_A(
            params.mainloop,
            load2transform_pipeline,
            load2transform_pipeline_producer_state,
            load_inputs,
            cta_coord_mnkl,
            k_tile_iter_next, k_tile_count - k_tile_prologue
          );
          load2transform_pipeline_producer_state = load2transform_pipeline_producer_state_next_;
        }

        if(is_participant.main_loadB){
          auto [load2mma_pipeline_producer_state_next, k_tile_iter_next] = collective_mainloop.load_B(
            params.mainloop,
            load2mma_pipeline,
            load2mma_pipeline_producer_state,
            load_inputs,
            cta_coord_mnkl,
            k_tile_iter, k_tile_prologue
          );
          load2mma_pipeline_producer_state = load2mma_pipeline_producer_state_next;

          auto [load2mma_pipeline_producer_state_next_, unused_] = collective_mainloop.load_B(
            params.mainloop,
            load2mma_pipeline,
            load2mma_pipeline_producer_state,
            load_inputs,
            cta_coord_mnkl,
            k_tile_iter_next, k_tile_count - k_tile_prologue
          );
          load2mma_pipeline_producer_state = load2mma_pipeline_producer_state_next_;

        }
      }

      // Sync warp to prevent non-participating threads entering next wave early
      __syncwarp();

      // Fetch next work tile
      auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
        work_tile_info,
        clc_pipeline,
        clc_pipeline_consumer_state
      );

      requires_clc_query = increment_pipe;
      if (increment_pipe) {
        ++clc_pipeline_consumer_state;
      }
      work_tile_info = next_work_tile_info;
    } while (work_tile_info.is_valid());

    if(is_participant.main_loadA){
      if (lane_predicate) {
        load2transform_pipeline.producer_tail(load2transform_pipeline_producer_state);
      }
    }
    if(is_participant.main_loadB){
      if (lane_predicate) {
        load2mma_pipeline.producer_tail(load2mma_pipeline_producer_state);
      }
    }
  }

  // Scheduler warp of the cluster's first CTA: issues the CLC queries, throttled by the
  // A-load warp, and drains the CLC pipeline.
  CUTLASS_DEVICE void
  run_sched(TileScheduler& scheduler, typename TileScheduler::WorkTileInfo work_tile_info,
            CLCPipeline& clc_pipeline, CLCPipelineState clc_pipeline_producer_state,
            CLCPipelineState clc_pipeline_consumer_state,
            CLCThrottlePipeline& clc_throttle_pipeline,
            CLCThrottlePipelineState clc_pipe_throttle_consumer_state,
            arch::ClusterBarrier& epilogue_throttle_barrier) {
    // Signal the epilogue warps to proceed once the prologue is complete
    epilogue_throttle_barrier.arrive();

    if constexpr (IsSchedDynamicPersistent) {
      // Whether a new CLC query must be performed.
      // See comment below where this variable is updated for a description of
      // why this variable is needed.
      bool requires_clc_query = true;

      cutlass::arch::wait_on_dependent_grids();

      do {
        if (requires_clc_query) {
          // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
          clc_throttle_pipeline.consumer_wait(clc_pipe_throttle_consumer_state);
          clc_throttle_pipeline.consumer_release(clc_pipe_throttle_consumer_state);
          ++clc_pipe_throttle_consumer_state;

          // Query next clcID and update producer state
          clc_pipeline_producer_state = scheduler.advance_to_next_work(
            clc_pipeline,
            clc_pipeline_producer_state
          );
       }

        // Fetch next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
          work_tile_info,
          clc_pipeline,
          clc_pipeline_consumer_state
        );

        // Only perform a new CLC query if we consumed a new CLC query result in
        // `fetch_next_work`. An example of a case in which CLC `fetch_next_work` does
        // not consume a new CLC query response is when processing stream-K units.
        // The current stream-K scheduler uses single WorkTileInfo to track multiple
        // (potentially-partial) tiles to be computed via stream-K. In this case,
        // `fetch_next_work` simply performs in-place updates on the existing WorkTileInfo,
        // rather than consuming a CLC query response.
        requires_clc_query = increment_pipe;
        if (increment_pipe) {
          ++clc_pipeline_consumer_state;
        }

        work_tile_info = next_work_tile_info;
      } while (work_tile_info.is_valid());
      clc_pipeline.producer_tail(clc_pipeline_producer_state);
    }
  }

  // Transform warpgroups: convert each k-tile of A from the load pipeline into the MMA
  // pipeline, then drain it.
  template <class ProblemShapeMNKL, class TensorA, class BulkTmem, class TransformInputs>
  CUTLASS_DEVICE void
  run_transform(ProblemShapeMNKL const& problem_shape_MNKL, TensorA const& gA_mkl,
                CollectiveMainloop& collective_mainloop, BulkTmem const& bulk_tmem,
                TransformInputs const& transform_inputs, TileScheduler& scheduler,
                typename TileScheduler::WorkTileInfo work_tile_info,
                Load2TransformPipeline& load2transform_pipeline,
                Load2TransformPipelineState load2transform_pipeline_consumer_state,
                Transform2MmaPipeline& transform2mma_pipeline,
                Transform2MmaPipelineState transform2mma_pipeline_producer_state,
                CLCPipeline& clc_pipeline, CLCPipelineState clc_pipeline_consumer_state,
                arch::NamedBarrier& tmem_allocation_result_barrier,
                arch::ClusterBarrier& epilogue_throttle_barrier) {
    using namespace cute;

    // Signal the epilogue warps to proceed once the prologue is complete
    epilogue_throttle_barrier.arrive();

    // Wait for tmem allocation
    tmem_allocation_result_barrier.arrive_and_wait_unaligned();

    do {
      auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
      auto k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
      auto k_tile_iter = cute::make_coord_iterator(idx2crd(k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));
      auto [load2transform_pipeline_consumer_state_next, transform2mma_pipeline_producer_state_next] = collective_mainloop.transform(
        load2transform_pipeline,
        load2transform_pipeline_consumer_state,
        transform2mma_pipeline,
        transform2mma_pipeline_producer_state,
        bulk_tmem,
        transform_inputs,
        k_tile_iter, k_tile_count
      );
      transform2mma_pipeline_producer_state = transform2mma_pipeline_producer_state_next;
      load2transform_pipeline_consumer_state = load2transform_pipeline_consumer_state_next;

      // Fetch next work tile
      auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
        work_tile_info,
        clc_pipeline,
        clc_pipeline_consumer_state
      );
      work_tile_info = next_work_tile_info;

      if (increment_pipe) {
        ++clc_pipeline_consumer_state;
      }
    } while (work_tile_info.is_valid());

    transform2mma_pipeline.producer_tail(transform2mma_pipeline_producer_state);
  }

  // MMA warp: owns the TMEM allocation for the CTA (and its release with the peer CTA), and
  // on the leader CTA issues the MMAs of every work tile.
  template <class ProblemShapeMNKL, class BulkTmem>
  CUTLASS_DEVICE void
  run_mma(SharedStorage& shared_storage, ProblemShapeMNKL const& problem_shape_MNKL,
          CollectiveMainloop& collective_mainloop, BulkTmem const& bulk_tmem, TileScheduler& scheduler,
          typename TileScheduler::WorkTileInfo work_tile_info, bool is_mma_leader_cta,
          uint32_t mma_peer_cta_rank, uint32_t dealloc_barrier_phase,
          TmemAllocator& tmem_allocator, arch::NamedBarrier& tmem_allocation_result_barrier,
          arch::ClusterBarrier& tmem_deallocation_result_barrier,
          arch::ClusterBarrier& epilogue_throttle_barrier,
          Load2MmaPipeline& load2mma_pipeline, Load2MmaPipelineState load2mma_pipeline_consumer_state,
          Transform2MmaPipeline& transform2mma_pipeline,
          Transform2MmaPipelineState transform2mma_pipeline_consumer_state,
          Mma2AccumPipeline& mma2accum_pipeline, Mma2AccumPipelineState mma2accum_pipeline_producer_state,
          CLCPipeline& clc_pipeline, CLCPipelineState clc_pipeline_consumer_state) {
    constexpr bool has_mma_peer_cta = cute::size(AtomThrShapeMNK{}) == 2;

    // Tmem allocation sequence
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
    __syncwarp();
    tmem_allocation_result_barrier.arrive();
    uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;

    auto mma_input_operands = collective_mainloop.mma_init(bulk_tmem, shared_storage.tensors.mainloop);

    // Signal the epilogue warps to proceed once the prologue is complete
    epilogue_throttle_barrier.arrive();

    do {
      auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
      // Fetch next work tile
      auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
        work_tile_info,
        clc_pipeline,
        clc_pipeline_consumer_state
      );
      work_tile_info = next_work_tile_info;

      if (increment_pipe) {
        ++clc_pipeline_consumer_state;
      }

      if (is_mma_leader_cta) {
          auto [load2mma_pipeline_consumer_state_next, transform2mma_pipeline_consumer_state_next, mma2accum_pipeline_producer_state_next] = collective_mainloop.mma(
            load2mma_pipeline,
            load2mma_pipeline_consumer_state,
            transform2mma_pipeline,
            transform2mma_pipeline_consumer_state,
            mma2accum_pipeline,
            mma2accum_pipeline_producer_state,
            bulk_tmem,
            mma_input_operands,
            k_tile_count
          );
          // Advance the mm2accum pipe
          load2mma_pipeline_consumer_state = load2mma_pipeline_consumer_state_next;
          transform2mma_pipeline_consumer_state = transform2mma_pipeline_consumer_state_next;
          mma2accum_pipeline_producer_state = mma2accum_pipeline_producer_state_next;
      }
    } while (work_tile_info.is_valid());

    // leader MMA waits for leader + peer epilogues to release accumulator stage
    if (is_mma_leader_cta) {
      mma2accum_pipeline.producer_tail(mma2accum_pipeline_producer_state);
    }

    // Hint on an early release of global memory resources.
    // The timing of calling this function only influences performance,
    // not functional correctness.
    cutlass::arch::launch_dependent_grids();

    // Signal to peer MMA that entire tmem allocation can be deallocated
    if constexpr (has_mma_peer_cta) {
      // Leader does wait + arrive, follower does arrive + wait
      tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, not is_mma_leader_cta);
      tmem_deallocation_result_barrier.wait(dealloc_barrier_phase);
      tmem_deallocation_result_barrier.arrive(mma_peer_cta_rank, is_mma_leader_cta);
    }

    // Free entire tmem allocation
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }

  // Epilogue-load warp (only when the epilogue reads C): loads the source tile of every
  // work tile that computes the epilogue, after the first A load has been issued.
  template <class ProblemShapeMNKL, class CtaCoordMNKL>
  CUTLASS_DEVICE void
  run_epilogue_load(Params const& params, SharedStorage& shared_storage,
                    ProblemShapeMNKL const& problem_shape_MNKL, CollectiveEpilogue& collective_epilogue,
                    TileScheduler& scheduler, typename TileScheduler::WorkTileInfo work_tile_info,
                    CtaCoordMNKL cta_coord_mnkl,
                    EpiLoadPipeline& epi_load_pipeline, EpiLoadPipelineState epi_load_pipe_producer_state,
                    EpiStorePipeline& epi_store_pipeline, EpiStorePipelineState epi_store_pipe_producer_state,
                    CLCPipeline& clc_pipeline, CLCPipelineState clc_pipeline_consumer_state,
                    LoadOrderBarrier& load_order_barrier, arch::ClusterBarrier& epilogue_throttle_barrier) {
    // Ensure that the prefetched kernel does not touch
    // unflushed global memory prior to this instruction
    cutlass::arch::wait_on_dependent_grids();

    bool do_load_order_wait = true;
    bool do_tail_load = false;

    // Signal the epilogue warps to proceed once the prologue is complete
    epilogue_throttle_barrier.arrive();

    do {
      bool compute_epilogue = TileScheduler::compute_epilogue(work_tile_info, params.scheduler);
      // Get current work tile and fetch next work tile
      auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
        work_tile_info,
        clc_pipeline,
        clc_pipeline_consumer_state
      );
      work_tile_info = next_work_tile_info;

      if (increment_pipe) {
        ++clc_pipeline_consumer_state;
      }

      if (compute_epilogue) {
        if (do_load_order_wait) {
          load_order_barrier.wait();
          do_load_order_wait = false;
        }

        epi_load_pipe_producer_state = collective_epilogue.load(
          epi_load_pipeline,
          epi_load_pipe_producer_state,
          problem_shape_MNKL,
          CtaShape_MNK{},
          cta_coord_mnkl,
          TileShape{},
          TiledMma{},
          shared_storage.tensors.epilogue
        );

        do_tail_load = true;
      }

      // Calculate the cta coordinates of the next work tile
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
    } while (work_tile_info.is_valid());

    // Only perform a tail load if one of the work units processed performed
    // an epilogue load. An example of a case in which a tail load should not be
    // performed is in split-K if a cluster is only assigned non-final splits (for which
    // the cluster does not compute the epilogue).
    if (do_tail_load) {
      collective_epilogue.load_tail(
        epi_load_pipeline, epi_load_pipe_producer_state,
        epi_store_pipeline, epi_store_pipe_producer_state);
    }
  }

  // Epilogue warps: for every work tile, fetch the next one first, then wait for the
  // accumulator stage, run the scheduler's fix-up, store the tile when this CTA computes
  // its epilogue, and only then install the next tile's coordinates.
  template <class ProblemShapeMNKL, class BulkTmem, class CtaCoordMNKL>
  CUTLASS_DEVICE void
  run_epilogue_store(SharedStorage& shared_storage, ProblemShapeMNKL const& problem_shape_MNKL,
                     CollectiveMainloop& collective_mainloop, CollectiveEpilogue& collective_epilogue,
                     BulkTmem const& bulk_tmem, TileScheduler& scheduler,
                     typename TileScheduler::WorkTileInfo work_tile_info, CtaCoordMNKL cta_coord_mnkl,
                     EpiLoadPipeline& epi_load_pipeline, EpiLoadPipelineState epi_load_pipe_consumer_state,
                     EpiStorePipeline& epi_store_pipeline, EpiStorePipelineState epi_store_pipe_producer_state,
                     Mma2AccumPipeline& mma2accum_pipeline,
                     Mma2AccumPipelineState mma2accum_pipeline_consumer_state,
                     CLCPipeline& clc_pipeline, CLCPipelineState clc_pipeline_consumer_state,
                     arch::NamedBarrier& tmem_allocation_result_barrier,
                     arch::ClusterBarrier& epilogue_throttle_barrier) {
    using namespace cute;

    // Throttle the epilogue warps to improve prologue performance
    static constexpr int epilogue_throttle_phase_bit = 0;
    epilogue_throttle_barrier.wait(epilogue_throttle_phase_bit);

    // Wait for tmem allocation
    tmem_allocation_result_barrier.arrive_and_wait_unaligned();

    auto accum_inputs = collective_mainloop.accum_init(bulk_tmem, typename CollectiveEpilogue::CopyOpT2R{}, typename CollectiveEpilogue::EpilogueTile{});
    bool do_tail_store = false;
    do {
      // Fetch next work tile
      auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
        work_tile_info,
        clc_pipeline,
        clc_pipeline_consumer_state
      );

      if (increment_pipe) {
        ++clc_pipeline_consumer_state;
      }

      auto k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, CtaShape_MNK{});
      mma2accum_pipeline.consumer_wait(mma2accum_pipeline_consumer_state);

      // Accumulators
      Tensor accumulators = bulk_tmem(_,_,_,mma2accum_pipeline_consumer_state.index()); // ((MMA_TILE_M,MMA_TILE_N),MMA_M,MMA_N)

      mma2accum_pipeline_consumer_state = scheduler.template fixup<IsComplex>(
        TiledMma{},
        work_tile_info,
        accumulators,
        mma2accum_pipeline,
        mma2accum_pipeline_consumer_state,
        typename CollectiveEpilogue::CopyOpT2R{}
      );

      //
      // Epilogue and write to gD
      //
      if (scheduler.compute_epilogue(work_tile_info)) {
        auto [load_state_next, store_state_next, mma2accum_pipeline_state_next] = collective_epilogue.store(
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          epi_store_pipeline,
          epi_store_pipe_producer_state,
          mma2accum_pipeline,
          mma2accum_pipeline_consumer_state,
          problem_shape_MNKL,
          CtaShape_MNK{},
          cta_coord_mnkl,
          TileShape{},
          TiledMma{},
          accumulators,
          shared_storage.tensors.epilogue
        );
        epi_load_pipe_consumer_state = load_state_next;
        epi_store_pipe_producer_state = store_state_next;
        do_tail_store = true;

        // Advance the mma2accum pipe
        mma2accum_pipeline_consumer_state = mma2accum_pipeline_state_next;
      }

      work_tile_info = next_work_tile_info;
      cta_coord_mnkl = scheduler.work_tile_to_cta_coord(work_tile_info);
    } while (work_tile_info.is_valid());

    // Only drain the epilogue store pipeline if one of the work units processed ran the
    // epilogue.  Under stream-K a cluster assigned only non-final K fragments of its
    // tiles never runs it, and then there is nothing to drain.
    if (do_tail_store) {
      collective_epilogue.store_tail(
        epi_load_pipeline, epi_load_pipe_consumer_state,
        epi_store_pipeline, epi_store_pipe_producer_state,
        CtaShape_MNK{});
    }
  }

  //
  // Pipeline parameters.  Each warp's role in a pipeline follows from its category; the
  // arrival counts and initializing warps are the kernel's fixed wiring.
  //

  static CUTLASS_DEVICE typename Load2TransformPipeline::Params
  load2transform_params(WarpCategory warp_category, bool is_leader) {
    typename Load2TransformPipeline::Params p;
    if (warp_category == WarpCategory::MainloopLoad) {
      p.role = Load2TransformPipeline::ThreadCategory::Producer;
    }
    else if (warp_category == WarpCategory::Transformation) {
      p.role = Load2TransformPipeline::ThreadCategory::Consumer;
    }
    p.is_leader = is_leader;
    p.num_consumers = NumTransformationThreads;
    p.transaction_bytes = CollectiveMainloop::TmaTransactionBytes_A;
    p.initializing_warp = 0;
    return p;
  }

  static CUTLASS_DEVICE typename Load2MmaPipeline::Params
  load2mma_params(WarpCategory warp_category, bool is_leader) {
    typename Load2MmaPipeline::Params p;
    if (warp_category == WarpCategory::MainloopLoadB) {
      p.role = Load2MmaPipeline::ThreadCategory::Producer;
    }
    else if (warp_category == WarpCategory::MMA) {
      p.role = Load2MmaPipeline::ThreadCategory::Consumer;
    }
    p.is_leader = is_leader;
    p.num_consumers = NumMMAThreads;
    p.transaction_bytes = CollectiveMainloop::TmaTransactionBytes_B;
    p.initializing_warp = 8;
    return p;
  }

  static CUTLASS_DEVICE typename Transform2MmaPipeline::Params
  transform2mma_params(WarpCategory warp_category) {
    typename Transform2MmaPipeline::Params p;
    if (warp_category == WarpCategory::Transformation) {
      p.role = Transform2MmaPipeline::ThreadCategory::Producer;
    }
    else if (warp_category == WarpCategory::MMA) {
      p.role = Transform2MmaPipeline::ThreadCategory::Consumer;
    }
    p.consumer_arv_count = 1;
    p.producer_arv_count = cute::size(AtomThrShapeMNK{}) * NumTransformationThreads;
    p.initializing_warp = 2;
    return p;
  }

  static CUTLASS_DEVICE typename Mma2AccumPipeline::Params
  mma2accum_params(WarpCategory warp_category) {
    typename Mma2AccumPipeline::Params p;
    if (warp_category == WarpCategory::MMA) {
      p.role = Mma2AccumPipeline::ThreadCategory::Producer;
    }
    else if (warp_category == WarpCategory::Epilogue) {
      p.role = Mma2AccumPipeline::ThreadCategory::Consumer;
    }
    p.producer_arv_count = 1;
    p.consumer_arv_count = cute::size(AtomThrShapeMNK{}) * NumEpilogueThreads;
    p.initializing_warp = 6;
    return p;
  }

  static CUTLASS_DEVICE typename EpiLoadPipeline::Params
  epi_load_params(WarpCategory warp_category, int cta_rank_in_cluster) {
    typename EpiLoadPipeline::Params p;
    if (WarpCategory::EpilogueLoad == warp_category) {
      p.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Epilogue == warp_category) {
      p.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    p.dst_blockid = cta_rank_in_cluster;
    p.producer_arv_count = NumEpilogueLoadThreads;
    p.consumer_arv_count = NumEpilogueThreads;
    p.transaction_bytes = CollectiveEpilogue::TmaTransactionBytes;
    p.initializing_warp = 4;
    return p;
  }

  static CUTLASS_DEVICE typename LoadOrderBarrier::Params
  load_order_params(WarpCategory warp_category) {
    typename LoadOrderBarrier::Params p;
    p.group_id = (warp_category == WarpCategory::MainloopLoad) ? 0 : 1;
    p.group_size = 1;
    p.initializing_warp = 5;
    return p;
  }

  static CUTLASS_DEVICE typename CLCPipeline::Params
  clc_params(WarpCategory warp_category, int cluster_size, bool is_epi_load_needed) {
    typename CLCPipeline::Params p;
    if (WarpCategory::Sched == warp_category) {
      p.role = CLCPipeline::ThreadCategory::ProducerConsumer;
    }
    else {
      p.role = CLCPipeline::ThreadCategory::Consumer;
    }
    p.producer_blockid = 0;
    p.producer_arv_count = 1;
    p.consumer_arv_count = NumSchedThreads + cluster_size *
                               (NumMainloopLoadThreads + NumMainloopLoadBThreads + NumEpilogueThreads +
                                NumMMAThreads + NumTransformationThreads);
    if (is_epi_load_needed) {
      p.consumer_arv_count += cluster_size * NumEpilogueLoadThreads;
    }
    p.transaction_bytes = CLCResponseSize;
    p.initializing_warp = 1;
    return p;
  }

  static CUTLASS_DEVICE typename CLCThrottlePipeline::Params
  clc_throttle_params(WarpCategory warp_category) {
    typename CLCThrottlePipeline::Params p;
    if (WarpCategory::MainloopLoad == warp_category) {
      p.role = CLCThrottlePipeline::ThreadCategory::Producer;
    }
    if (WarpCategory::Sched == warp_category) {
      p.role = CLCThrottlePipeline::ThreadCategory::Consumer;
    }
    p.producer_arv_count = NumMainloopLoadThreads;
    p.consumer_arv_count = NumSchedThreads;
    p.dst_blockid = 0;
    p.initializing_warp = 3;
    return p;
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
