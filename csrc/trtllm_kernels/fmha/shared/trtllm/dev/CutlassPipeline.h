/***************************************************************************************************
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cuda_ptx/cuda_ptx.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm_coord.h>

#include <cute/layout.hpp>
#include <cutlass/gemm/kernel/sm100_tile_scheduler.hpp>
#include <cutlass/gemm/kernel/sm90_tile_scheduler.hpp>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include "Utils.h"

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// PipelineTmaMultiUmmaAsync class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

// This class customizes cutlass::PipelineTmaUmmaAsync to allow for multiple UMMA consumers.
// The TMA producer is consumed by more than one UMMA consumers.
template <int Stages_, cutlass::McastDirection mcastDirection = cutlass::McastDirection::kRowCol,
          class ClusterShape = cute::Shape<int, int, cute::_1>,
          class AtomThrShape_MNK_ = cute::Shape<cute::_1, cute::_1, cute::_1>>
class PipelineTmaMultiUmmaAsync
    : public cutlass::PipelineTmaUmmaAsync<Stages_, ClusterShape, AtomThrShape_MNK_> {
 public:
  static constexpr uint32_t Stages = Stages_;
  using AtomThrShape_MNK = AtomThrShape_MNK_;

 private:
  using Base = cutlass::PipelineTmaUmmaAsync<Stages_, ClusterShape, AtomThrShape_MNK_>;

 public:
  using FullBarrier = typename Base::FullBarrier;
  using EmptyBarrier = typename Base::EmptyBarrier;
  using ProducerBarrierType = typename Base::ProducerBarrierType;
  using ConsumerBarrierType = typename Base::ConsumerBarrierType;
  using PipelineState = typename Base::PipelineState;
  using SharedStorage = typename Base::SharedStorage;
  using ThreadCategory = typename Base::ThreadCategory;
  using Params = typename Base::Params;

  // In case the pipeline is being used for a single side
  using McastDirection = cutlass::McastDirection;

  // Helper function to initialize barriers.
  static CUTLASS_DEVICE void init_barriers(SharedStorage& storage, int warpId, Params params,
                                           ClusterShape cluster_shape) {
    if (warpId == params.initializing_warp) {
      constexpr int producer_arv_cnt = 1;
      auto atom_thr_shape = AtomThrShape_MNK{};

      uint32_t multicast_consumer_arrival_count;
      if constexpr (mcastDirection == cutlass::McastDirection::kRowCol) {
        multicast_consumer_arrival_count =
            (cute::size<0>(cluster_shape) / cute::size<0>(atom_thr_shape)) +
            (cute::size<1>(cluster_shape) / cute::size<1>(atom_thr_shape)) - 1;
      } else if constexpr (mcastDirection == cutlass::McastDirection::kRow) {
        multicast_consumer_arrival_count =
            cute::size<1>(cluster_shape) / cute::size<1>(atom_thr_shape);
      } else {
        multicast_consumer_arrival_count =
            cute::size<0>(cluster_shape) / cute::size<0>(atom_thr_shape);
      }

      cutlass::arch::detail::initialize_barrier_array_pair_aligned<
          decltype(storage.full_barrier_), decltype(storage.empty_barrier_), Stages>(
          storage.full_barrier_, storage.empty_barrier_, producer_arv_cnt,
          multicast_consumer_arrival_count * params.num_consumers);  // Multi-UMMA consumers.
    }
  }

  // Constructor by default initializes barriers and calculates masks.
  // These operations can be explicity deferred by specifying InitBarriers and InitMasks.
  // If deferred, user code needs to guarantee init_masks and/or init_barriers is/are called.
  template <typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  CUTLASS_DEVICE PipelineTmaMultiUmmaAsync(SharedStorage& storage, int32_t warpId, Params params,
                                           ClusterShape cluster_shape, InitBarriers = {},
                                           InitMasks = {})
      : Base(storage, params, cluster_shape, cute::false_type{}, cute::false_type{}) {
    static_assert(cute::is_same_v<InitBarriers, cute::true_type> ||
                  cute::is_same_v<InitBarriers, cute::false_type>);
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      init_barriers(storage, warpId, params, cluster_shape);
    }

    static_assert(cute::is_same_v<InitMasks, cute::true_type> ||
                  cute::is_same_v<InitMasks, cute::false_type>);
    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      if constexpr (mcastDirection == McastDirection::kRowCol) {
        Base::init_masks(cluster_shape);
      } else {
        Base::init_masks(cluster_shape, mcastDirection);
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// StaticPersistentPipelinedTileSchedulerSm90 class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

// Pipelined (work id in ring buffer) Static Persistent scheduler for SM90.
// Uses a producer-consumer pipeline where the producer computes work IDs
// ahead of time and stores them in shared memory ring buffer.
// Consumers read work IDs from shared memory instead of computing individually.
// Inherits from StaticPersistentTileScheduler via CRTP to reuse most functionality.
template <class ClusterShape_, uint32_t Stages_>
class StaticPersistentPipelinedTileSchedulerSm90
    : public cutlass::gemm::kernel::detail::StaticPersistentTileScheduler<
          StaticPersistentPipelinedTileSchedulerSm90<ClusterShape_, Stages_>> {
 private:
  using Base = cutlass::gemm::kernel::detail::StaticPersistentTileScheduler<
      StaticPersistentPipelinedTileSchedulerSm90<ClusterShape_, Stages_>>;
  using Self = StaticPersistentPipelinedTileSchedulerSm90<ClusterShape_, Stages_>;
  using Sm90Scheduler = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90;

 public:
  using typename Base::Arguments;
  using typename Base::Params;
  using typename Base::RasterOrder;

  using ClusterShape = ClusterShape_;
  static constexpr uint32_t ClusterSize = cute::size(ClusterShape_{});
  static constexpr uint32_t Stages = Stages_;

  // Work ID response stored in shared memory ring buffer
  using WorkTileInfo =
      typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::WorkTileInfo;
  using CLCResponse =
      typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClusterShape,
                                                                           Stages>::CLCResponse;

  // Pipeline types for producer-consumer synchronization
  using Pipeline = cutlass::PipelineCLCFetchAsync<Stages, ClusterShape>;
  using SharedStorage = typename Pipeline::SharedStorage;

  static CUTLASS_DEVICE cute::tuple<int32_t, int32_t> get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim, cutlass::FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      cutlass::FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      cutlass::FastDivmodU64 const& divmod_cluster_blk_major, int32_t log_swizzle_size,
      RasterOrder raster_order) {
    return Sm90Scheduler::get_work_idx_m_and_n(blk_per_grid_dim, divmod_cluster_shape_major,
                                               divmod_cluster_shape_minor, divmod_cluster_blk_major,
                                               log_swizzle_size, raster_order);
  }

  // Ctor.
  CUTLASS_DEVICE
  explicit StaticPersistentPipelinedTileSchedulerSm90(Params const& params) : Base(params) {}

  CUTLASS_DEVICE explicit StaticPersistentPipelinedTileSchedulerSm90(CLCResponse* clc_response_ptr,
                                                                     Params const& params,
                                                                     dim3 block_id_in_cluster)
      : Base(params),
        clc_response_ptr_(clc_response_ptr),
        block_id_in_cluster_(block_id_in_cluster) {}

  // Returns invalid as the first tile will be fetched via fetch_next_work.
  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape /* cluster_shape */) {
    // Return invalid tile since work will be consumed via fetch_next_work
    return WorkTileInfo::invalid_work_tile();
  }

  // Producer API. Compute work ID and store in shared memory.
  CUTLASS_DEVICE
  cutlass::PipelineState<Stages> advance_to_next_work(
      Pipeline& pipeline, cutlass::PipelineState<Stages> pipe_producer_state,
      int32_t advance_count = 1) const {
    auto non_const_this = const_cast<Base*>(static_cast<Base const*>(this));
    // Advance the linear index.
    non_const_this->advance_to_next_work(advance_count);
    // Calculate work id from linear index.
    typename Base::WorkTileInfo work_id = Base::get_current_work();
    // Wait for buffer slot to become empty.
    pipeline.producer_acquire(pipe_producer_state);
    // Convert WorkTileInfo struct to 16B CLCResponse.
    CLCResponse clc_response;
    clc_response.data[0] = work_id.M_idx;
    clc_response.data[1] = work_id.N_idx;
    clc_response.data[2] = work_id.L_idx;
    clc_response.data[3] = static_cast<uint32_t>(work_id.is_valid_tile);
    // Store work ID in shared memory ring buffer.
    if constexpr (ClusterSize == 1) {
      // Use direct store + manual barrier completion instead.
      clc_response_ptr_[pipe_producer_state.index()] = clc_response;
      // Fence between SM and SYNCS.
      cutlass::arch::fence_view_async_shared();
      // Manually complete the transaction bytes that producer_acquire set up.
      pipeline.producer_commit(pipe_producer_state);
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (uint32_t rank = 0; rank < ClusterSize; ++rank) {
        uint32_t* current_clc_response_ptr =
            reinterpret_cast<uint32_t*>(&clc_response_ptr_[pipe_producer_state.index()]);
        uint32_t* remote_addr =
            cuda_ptx::mapa(cuda_ptx::space_cluster, current_clc_response_ptr, rank);
        uint64_t* remote_bar = cuda_ptx::mapa(
            cuda_ptx::space_cluster,
            reinterpret_cast<uint64_t*>(
                __cvta_shared_to_generic(pipeline.producer_get_barrier(pipe_producer_state))),
            rank);
        if (cute::elect_one_sync()) {
          cuda_ptx::st_async(remote_addr, clc_response.data, remote_bar);
        }
      }
    }
    // Advance the pipeline state.
    ++pipe_producer_state;
    // Return the new pipeline state.
    return pipe_producer_state;
  }

  // Consumer API. Read work ID from shared memory.
  template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo /* work_tile_info */,
                                      TileSchedulerPipeline& scheduler_pipeline,
                                      TileSchedulerPipelineState scheduler_pipe_consumer_state) {
    // Wait for producer to commit data.
    scheduler_pipeline.consumer_wait(scheduler_pipe_consumer_state);
    // Read work ID from shared memory.
    CLCResponse clc_response = clc_response_ptr_[scheduler_pipe_consumer_state.index()];
    // Release the buffer slot for producer.
    scheduler_pipeline.consumer_release(scheduler_pipe_consumer_state);
    // Convert to WorkTileInfo.
    WorkTileInfo new_work_tile;
    new_work_tile.M_idx = clc_response.data[0];
    new_work_tile.N_idx = clc_response.data[1];
    new_work_tile.L_idx = clc_response.data[2];
    new_work_tile.is_valid_tile = clc_response.data[3] == 1;
    // Add cluster offsets.
    new_work_tile.M_idx += block_id_in_cluster_.x;
    new_work_tile.N_idx += block_id_in_cluster_.y;
    // Return work tile and flag indicating pipeline state should advance.
    return cute::make_tuple(new_work_tile, true);
  }

 private:
  CLCResponse* clc_response_ptr_ = nullptr;
  dim3 block_id_in_cluster_ = {0, 0, 0};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Dynamic persistent tile scheduler that uses atomicAdd on a global counter for work distribution.
// This enables dynamic work-stealing among thread blocks, providing better load balancing
// when work tiles have varying execution times.
// Similar to StaticPersistentPipelinedTileSchedulerSm90 but uses atomic operations instead of
// static linear index increment.
template <class ClusterShape_, uint32_t Stages_>
class DynamicPersistentPipelinedTileSchedulerSm90
    : public cutlass::gemm::kernel::detail::StaticPersistentTileScheduler<
          DynamicPersistentPipelinedTileSchedulerSm90<ClusterShape_, Stages_>> {
 private:
  using Base = cutlass::gemm::kernel::detail::StaticPersistentTileScheduler<
      DynamicPersistentPipelinedTileSchedulerSm90<ClusterShape_, Stages_>>;
  using Self = DynamicPersistentPipelinedTileSchedulerSm90<ClusterShape_, Stages_>;
  using Sm90Scheduler = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90;

 public:
  using typename Base::Arguments;
  using typename Base::Params;
  using typename Base::RasterOrder;

  using ClusterShape = ClusterShape_;
  static constexpr uint32_t ClusterSize = cute::size(ClusterShape_{});
  static constexpr uint32_t Stages = Stages_;

  // Work ID response stored in shared memory ring buffer
  using WorkTileInfo =
      typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::WorkTileInfo;
  using CLCResponse =
      typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100<ClusterShape,
                                                                           Stages>::CLCResponse;

  // Pipeline types for producer-consumer synchronization
  using Pipeline = cutlass::PipelineCLCFetchAsync<Stages, ClusterShape>;
  using SharedStorage = typename Pipeline::SharedStorage;

  static CUTLASS_DEVICE cute::tuple<int32_t, int32_t> get_work_idx_m_and_n(
      uint64_t blk_per_grid_dim, cutlass::FastDivmodU64Pow2 const& divmod_cluster_shape_major,
      cutlass::FastDivmodU64Pow2 const& divmod_cluster_shape_minor,
      cutlass::FastDivmodU64 const& divmod_cluster_blk_major, int32_t log_swizzle_size,
      RasterOrder raster_order) {
    return Sm90Scheduler::get_work_idx_m_and_n(blk_per_grid_dim, divmod_cluster_shape_major,
                                               divmod_cluster_shape_minor, divmod_cluster_blk_major,
                                               log_swizzle_size, raster_order);
  }

  // Ctor.
  CUTLASS_DEVICE
  explicit DynamicPersistentPipelinedTileSchedulerSm90(Params const& params) : Base(params) {}

  CUTLASS_DEVICE explicit DynamicPersistentPipelinedTileSchedulerSm90(CLCResponse* clc_response_ptr,
                                                                      Params const& params,
                                                                      dim3 block_id_in_cluster,
                                                                      uint32_t* dPtrTileCounter)
      : Base(params),
        clc_response_ptr_(clc_response_ptr),
        block_id_in_cluster_(block_id_in_cluster),
        dPtrTileCounter_(dPtrTileCounter),
        current_linear_idx_(0) {
    // Compute initial linear index same as the Base class does.
    if (params.raster_order_ == RasterOrder::AlongN) {
      current_linear_idx_ = uint64_t(blockIdx.x) + uint64_t(blockIdx.y) * uint64_t(gridDim.x);
    } else {
      current_linear_idx_ = uint64_t(blockIdx.x) * uint64_t(gridDim.y) + uint64_t(blockIdx.y);
    }
  }

  // Returns invalid as the first tile will be fetched via fetch_next_work.
  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape /* cluster_shape */) {
    // Return invalid tile since work will be consumed via fetch_next_work
    return WorkTileInfo::invalid_work_tile();
  }

  // Producer API. Atomically fetch next work index and store in shared memory.
  CUTLASS_DEVICE
  cutlass::PipelineState<Stages> advance_to_next_work(
      Pipeline& pipeline, cutlass::PipelineState<Stages> pipe_producer_state,
      int32_t advance_count = 1) const {
    // To bootstrap, get the current work id calculated from blockIdx to save one atomic operation.
    // To continue, fetch the next work id from the dynamic scheduler.
    if (advance_count > 0) {
      // Only one thread per cluster performs the atomic operation.
      if (cute::elect_one_sync()) {
        // Use atomicAdd to get the next linear work index dynamically.
        current_linear_idx_ = atomicAdd(dPtrTileCounter_, static_cast<uint32_t>(advance_count));
      }
      // Broadcast the linear index to all threads in the warp.
      current_linear_idx_ = __shfl_sync(0xFFFFFFFF, current_linear_idx_, 0);
    }
    // Calculate work id from the linear index.
    typename Base::WorkTileInfo work_id =
        Base::get_current_work_for_linear_idx(current_linear_idx_);
    // Wait for buffer slot to become empty.
    pipeline.producer_acquire(pipe_producer_state);
    // Convert WorkTileInfo struct to 16B CLCResponse.
    CLCResponse clc_response;
    clc_response.data[0] = work_id.M_idx;
    clc_response.data[1] = work_id.N_idx;
    clc_response.data[2] = work_id.L_idx;
    clc_response.data[3] = static_cast<uint32_t>(work_id.is_valid_tile);
    // Store work ID in shared memory ring buffer.
    if constexpr (ClusterSize == 1) {
      // Use direct store + manual barrier completion instead.
      clc_response_ptr_[pipe_producer_state.index()] = clc_response;
      // Fence between SM and SYNCS.
      cutlass::arch::fence_view_async_shared();
      // Manually complete the transaction bytes that producer_acquire set up.
      pipeline.producer_commit(pipe_producer_state);
    } else {
      CUTLASS_PRAGMA_UNROLL
      for (uint32_t rank = 0; rank < ClusterSize; ++rank) {
        uint32_t* current_clc_response_ptr =
            reinterpret_cast<uint32_t*>(&clc_response_ptr_[pipe_producer_state.index()]);
        uint32_t* remote_addr =
            cuda_ptx::mapa(cuda_ptx::space_cluster, current_clc_response_ptr, rank);
        uint64_t* remote_bar = cuda_ptx::mapa(
            cuda_ptx::space_cluster,
            reinterpret_cast<uint64_t*>(
                __cvta_shared_to_generic(pipeline.producer_get_barrier(pipe_producer_state))),
            rank);
        if (cute::elect_one_sync()) {
          cuda_ptx::st_async(remote_addr, clc_response.data, remote_bar);
        }
      }
    }
    // Advance the pipeline state.
    ++pipe_producer_state;
    // Return the new pipeline state.
    return pipe_producer_state;
  }

  // Consumer API. Read work ID from shared memory.
  template <class TileSchedulerPipeline, class TileSchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo /* work_tile_info */,
                                      TileSchedulerPipeline& scheduler_pipeline,
                                      TileSchedulerPipelineState scheduler_pipe_consumer_state) {
    // Wait for producer to commit data.
    scheduler_pipeline.consumer_wait(scheduler_pipe_consumer_state);
    // Read work ID from shared memory.
    CLCResponse clc_response = clc_response_ptr_[scheduler_pipe_consumer_state.index()];
    // Release the buffer slot for producer.
    scheduler_pipeline.consumer_release(scheduler_pipe_consumer_state);
    // Convert to WorkTileInfo.
    WorkTileInfo new_work_tile;
    new_work_tile.M_idx = clc_response.data[0];
    new_work_tile.N_idx = clc_response.data[1];
    new_work_tile.L_idx = clc_response.data[2];
    new_work_tile.is_valid_tile = clc_response.data[3] == 1;
    // Add cluster offsets.
    new_work_tile.M_idx += block_id_in_cluster_.x;
    new_work_tile.N_idx += block_id_in_cluster_.y;
    // Return work tile and flag indicating pipeline state should advance.
    return cute::make_tuple(new_work_tile, true);
  }

  // Get the current linear index that was fetched by atomicAdd.
  CUTLASS_DEVICE
  uint64_t get_current_linear_idx() const { return current_linear_idx_; }

 private:
  CLCResponse* clc_response_ptr_ = nullptr;
  dim3 block_id_in_cluster_ = {0, 0, 0};
  uint32_t* dPtrTileCounter_ = nullptr;
  uint64_t mutable current_linear_idx_ = 0;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::kernel::detail

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassPipelineState utility function.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PipelineState>
__device__ PipelineState makePipelineState(PipelineState state, int stateIncr) {
  return PipelineState::make_pipeline_state(state, stateIncr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PipelineState>
__device__ PipelineState makeProdStartStateFrom(PipelineState) {
  return PipelineState(0, 1, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PipelineState>
__device__ PipelineState makeConsStartStateFrom(PipelineState) {
  return PipelineState();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

// The active FlashInfer JIT toolchain cannot assemble the PTX 9.4 barrier modifiers used by the
// imported CGA pipeline on SM100 (`mbarrier.init.layout::v1` and the phase-typed
// `mbarrier.test_wait.parity` forms are rejected by ptxas). Use the newer forms only on the arch
// family that supports them natively; otherwise map to the older SM90/SM100-compatible barrier ops
// so the bundle still compiles while the launcher falls back to cubin for the unsupported CGA path.
template <typename = void>
CUTLASS_DEVICE void mbarrier_init_payload64_compat(std::uint64_t* addr, std::uint32_t count) {
#if __libcuda_ptx_isa >= 940 && (defined(_NVHPC_CUDA) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1070))
  cuda_ptx::mbarrier_init(cuda_ptx::layout_v1, addr, count);
#else
  cuda_ptx::mbarrier_init(addr, count);
#endif
}

template <cuda_ptx::dot_scope Scope>
CUTLASS_DEVICE bool mbarrier_test_wait_parity_primary_compat(cuda_ptx::sem_acquire_t,
                                                             cuda_ptx::scope_t<Scope> scope,
                                                             bool& isReportSeen,
                                                             std::uint64_t* addr,
                                                             std::uint32_t phaseParity) {
#if __libcuda_ptx_isa >= 940 && (defined(_NVHPC_CUDA) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1070))
  return cuda_ptx::mbarrier_test_wait_parity(cuda_ptx::mbarrier_phase_primary_t{},
                                             cuda_ptx::sem_acquire_t{}, scope, isReportSeen, addr,
                                             phaseParity);
#else
  isReportSeen = false;
  return cuda_ptx::mbarrier_test_wait_parity(cuda_ptx::sem_acquire_t{}, scope, addr, phaseParity);
#endif
}

template <cuda_ptx::dot_scope Scope>
CUTLASS_DEVICE bool mbarrier_test_wait_parity_conditional_compat(cuda_ptx::sem_acquire_t,
                                                                 cuda_ptx::scope_t<Scope> scope,
                                                                 std::uint64_t* addr,
                                                                 std::uint32_t phaseParity) {
#if __libcuda_ptx_isa >= 940 && (defined(_NVHPC_CUDA) || (_LIBCUDA_PTX_ARCH_SPECIFIC() == 1070))
  return cuda_ptx::mbarrier_test_wait_parity(cuda_ptx::mbarrier_phase_conditional_t{},
                                             cuda_ptx::sem_acquire_t{}, scope, addr, phaseParity);
#else
  return cuda_ptx::mbarrier_test_wait_parity(cuda_ptx::sem_acquire_t{}, scope, addr, phaseParity);
#endif
}

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassOrderedSequenceBarrier class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int SequenceDepth, int SequenceLength>
class CutlassOrderedSequenceBarrier {
  // The CUTLASS pipeline.
  using Pipeline = cutlass::OrderedSequenceBarrier<SequenceDepth, SequenceLength>;
  // The parameters to initialize the pipeline.
  using Params = typename Pipeline::Params;

 public:
  // The shared memory storage.
  using SharedStorage = typename Pipeline::SharedStorage;

 public:
  // The ctor.
  inline explicit __device__ CutlassOrderedSequenceBarrier(SharedStorage& sharedStorage, int warpId,
                                                           int group_id, int group_size,
                                                           int barInitWarpId = 0)
      : mParams{static_cast<uint32_t>(group_id), static_cast<uint32_t>(group_size), barInitWarpId},
        mPipeline(
            /* storage */ sharedStorage,
            /* params */ mParams) {
#if (__CUDA_ARCH__ >= 1000)
    if (warpId == barInitWarpId) {
      auto barrier_ptr = &sharedStorage.barrier_[0][0];
      int arv_cnt = mParams.group_size;
      constexpr int Stages = SequenceDepth * SequenceLength;
      cutlass::arch::detail::initialize_barrier_array_aligned<decltype(barrier_ptr), Stages>(
          barrier_ptr, arv_cnt);
    }
#else
    // Init is done only by the one elected thread of the block
    int lane_predicate = cute::elect_one_sync();
    if (warpId == barInitWarpId && lane_predicate) {
      for (int d = 0; d < SequenceDepth; ++d) {
        for (int l = 0; l < SequenceLength; ++l) {
          sharedStorage.barrier_[d][l].init(mParams.group_size);
        }
      }
    }
#endif
    // Note: fence_barrier_init() will be invoked once after all barriers are initialized.
  }

  // Signal completion of stage and move to the next stage
  inline __device__ void arrive() { mPipeline.arrive(); }

  // Wait on a stage to be unlocked
  inline __device__ void wait() { mPipeline.wait(); }

 private:
  Params mParams;
  Pipeline mPipeline;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassOrderedSequenceNamedBarrier class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

// SequenceDepth = 1
// SequenceLength = 2
class CutlassOrderedSequenceNamedBarrier {
 public:
  // The ctor.
  inline explicit __device__ CutlassOrderedSequenceNamedBarrier(int group_id, int group_size,
                                                                int namedBarId)
      : mGroupId{group_id}, mAllGroupsSize(group_size * 2), mNamedBarId(namedBarId) {}

  // Signal completion of stage and move to the next stage
  inline __device__ void arrive() {
    // If it is the 1st group
    if (mGroupId == 0) {
      asm volatile("bar.sync %0, %1;" ::"r"(mNamedBarId), "r"(mAllGroupsSize));
    }
  }

  // Wait on a stage to be unlocked
  inline __device__ void wait() {
    // If it is the second group (group id == 1) then wait for the first group to arrive
    // Otherwise (group id == 0) do nothing, just proceed unblocked
    if (mGroupId == 1) {
      asm volatile("bar.sync %0, %1;" ::"r"(mNamedBarId), "r"(mAllGroupsSize));
    }
  }

 private:
  int mGroupId;
  int mAllGroupsSize;
  int mNamedBarId;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassClusterBarrier class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>,
          bool SyncsTwoCtasForMma = false>
class CutlassClusterBarrier {
  // The CUTLASS barrier.
  using Barrier = cutlass::arch::ClusterBarrier;

 public:
  // The shared memory storage same as the pipeline.
  using SharedStorage = Barrier;

 public:
  // Ctor.
  inline __device__ CutlassClusterBarrier(SharedStorage& sharedStorage, int32_t warpId,
                                          int32_t numThreadsPerCta, int32_t barInitWarpId = 0,
                                          ClusterShape clusterShape = ClusterShape{})
      : mBarrierPtr{&sharedStorage}, mClusterShape{clusterShape} {
    // The lead CTA does wait + arrive, and the follower CTA does arrive + wait. Specifically:
    // 1. Follower CTAs arrives at lead CTA's barrier.
    // 2. Lead CTA waits for all follower CTAs to arrive.
    // 3. Lead CTA arrives at follower CTA's barrier one-by-one.
    // 4. Follower CTA waits for lead CTA to arrive.
    // Therefore, the expected arrival count for each role are:
    // The expected arrival count of the leader CTA is (CGA_size - 1) * arvCnt.
    // The expected arrival count of the follower CTAs is arvCnt.

    // The cluster size must be 2 if SyncsTwoCtasForMma is true.
    static_assert(!SyncsTwoCtasForMma || cute::size<0>(clusterShape) == 2, "Not supported");

    // The block rank in the cluster.
    int const blockRank = cute::block_rank_in_cluster();
    // The lead CTA's rank.
    int32_t const leadCtaRank = SyncsTwoCtasForMma ? (blockRank & 0xfffffffe) : 0;
    int producerArriveCount =
        SyncsTwoCtasForMma ? cute::size<0>(clusterShape) - 1 : cute::size(clusterShape) - 1;
    // The expected arrival count for the current CTA.
    int arriveCount = (blockRank == leadCtaRank ? producerArriveCount : 1) * numThreadsPerCta;
    // Elect one thread to initialize the barrier.
    if (cute::elect_one_sync() && warpId == barInitWarpId) {
      mBarrierPtr->init(arriveCount);
    }
  }

  // Single instruction to arrive and wait.
  inline __device__ void sync(int phase = 0) {
    int numRanksInCluster =
        (SyncsTwoCtasForMma ? cute::size<0>(mClusterShape) : cute::size(mClusterShape));
    // The block rank in the cluster.
    int32_t const blockRank = cute::block_rank_in_cluster();
    // The lead CTA's rank.
    int32_t const leadCtaRank = SyncsTwoCtasForMma ? (blockRank & 0xfffffffe) : 0;
    // Follower CTA arrives at lead CTA.
    mBarrierPtr->arrive(leadCtaRank, blockRank != leadCtaRank);
    // Lead CTA waits on follower CTAs. Follower CTAs waits for the lead CTA.
    mBarrierPtr->wait(phase);
    // Lead CTA arrives at follower CTA's barrier one-by-one.
    for (int ii = 1; ii < numRanksInCluster; ++ii) {
      mBarrierPtr->arrive(leadCtaRank + ii, blockRank == leadCtaRank);
    }
  }

  // Single instruction to arrive and wait.
  inline __device__ void sync(int phase, int numRanksInCluster) {
    // The block rank in the cluster.
    int32_t const blockRank = cute::block_rank_in_cluster();
    // The lead CTA's rank.
    int32_t const leadCtaRank = SyncsTwoCtasForMma ? (blockRank & 0xfffffffe) : 0;
    // Follower CTA arrives at lead CTA.
    mBarrierPtr->arrive(leadCtaRank, blockRank != leadCtaRank);
    // Arrive on the lead CTA's barrier to taken into account the unused ranks.
    if constexpr (!SyncsTwoCtasForMma) {
      for (int ii = numRanksInCluster;
           ii < (SyncsTwoCtasForMma ? cute::size<0>(mClusterShape) : cute::size(mClusterShape));
           ++ii) {
        mBarrierPtr->arrive(leadCtaRank, blockRank == leadCtaRank);
      }
    }
    // Lead CTA waits on follower CTAs. Follower CTAs waits for the lead CTA.
    mBarrierPtr->wait(phase);
    // Lead CTA arrives at follower CTA's barrier one-by-one.
    for (int ii = 1; ii < numRanksInCluster; ++ii) {
      mBarrierPtr->arrive(leadCtaRank + ii, blockRank == leadCtaRank);
    }
  }

  // Arrive on the barrier.
  inline __device__ void arrive(int ctaIdx, bool pred = true) { mBarrierPtr->arrive(ctaIdx, pred); }

  // Wait on the barrier.
  inline __device__ void wait(int phase) { mBarrierPtr->wait(phase); }

 private:
  // Declare the barrier's shared pointer.
  Barrier* mBarrierPtr;
  // The clusterShape
  ClusterShape mClusterShape;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassClusterTransactionBarrier class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>>
class CutlassClusterTransactionBarrier {
  // The CUTLASS barrier.
  using Barrier = cutlass::arch::ClusterTransactionBarrier;

 public:
  // The shared memory storage same as the pipeline.
  using SharedStorage = Barrier;

 public:
  // Ctor.
  inline __device__ CutlassClusterTransactionBarrier(SharedStorage& sharedStorage, int32_t warpId,
                                                     int32_t arriveCnt, uint32_t transactionBytes,
                                                     int32_t barInitWarpId = 0,
                                                     ClusterShape = ClusterShape{})
      : mBarrierPtr{&sharedStorage} {
    // Elect one thread to initialize the barrier and set the expected transaction bytes.
    if (warpId == barInitWarpId && cute::elect_one_sync()) {
      mBarrierPtr->init(arriveCnt);
      // Arrive and set the expected transaction bytes if provided.
      // This is meant to avoid additional cluster_sync.
      if (transactionBytes > 0) {
        mBarrierPtr->arrive_and_expect_tx(transactionBytes);
      }
    }
  }

  // Arrive and set the expected transaction bytes.
  inline __device__ void arrive_and_expect_tx(int ctaIdx, uint32_t transactionBytes,
                                              bool pred = true) {
    mBarrierPtr->arrive_and_expect_tx(ctaIdx, transactionBytes, pred);
  }

  // Complete the transaction.
  inline __device__ void complete_transaction(uint32_t dst_cta_id, uint32_t transaction_bytes,
                                              bool pred = true) {
    mBarrierPtr->complete_transaction(dst_cta_id, transaction_bytes, pred);
  }

  // Wait for the barrier.
  inline __device__ void wait(int phase = 0) { mBarrierPtr->wait(phase); }

  // Get the barrier pointer.
  inline __device__ uint64_t* getBarrierPtr() { return reinterpret_cast<uint64_t*>(mBarrierPtr); }

 private:
  // Declare the barrier's shared pointer.
  Barrier* mBarrierPtr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassCpAsyncPipeline class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

// UseCpAsyncWaitGroup changes the fencing of cp.async wrt other proxies:
//   TRUE: user manually inserts cp.async.commit_group (LDGDEPBAR) before barrier arrival.
//  FALSE: codegen auto-inserts cp.async.mbarrier.arrive (ARRIVES.LDGSTSBAR) before barrier arrival.
// In other words, when UseCpAsyncWaitGroup == true, this acts as simple arrive/wait pipeline.
template <int32_t NumStages, bool UseCpAsyncWaitGroup = false>
class CutlassCpAsyncPipeline {
  // The CUTLASS pipeline.
  using Pipeline = cutlass::PipelineAsync<NumStages>;
  // The parameters to initialize the pipeline.
  using Params = typename Pipeline::Params;

 public:
  // The state.
  using PipelineState = typename Pipeline::PipelineState;
  // The shared memory storage.
  using SharedStorage = typename Pipeline::SharedStorage;

 public:
  // Ctor. We may want to use mPipeline{sharedStorage, {}, cute::false_type{}} in the future.
  inline explicit __device__ CutlassCpAsyncPipeline(SharedStorage& sharedStorage, int32_t warpId,
                                                    int32_t prodArvCnt = 1, int32_t consArvCnt = 1,
                                                    int32_t barInitWarpId = 0)
      : mPipeline{sharedStorage,
                  Params{Pipeline::ThreadCategory::ProducerConsumer,
                         reinterpret_cast<uint32_t const&>(prodArvCnt),
                         reinterpret_cast<uint32_t const&>(consArvCnt),
                         cute::block_rank_in_cluster() /* dst_block_id */, barInitWarpId},
                  cute::false_type{}} {
    // Initialize barriers. Doing this here instead of in cutlass avoids unnecessary shuffle of warp
    // id for each pipeline and redundant predicates and branches.
    if (warpId == barInitWarpId) {
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<
          decltype(sharedStorage.full_barrier_), decltype(sharedStorage.empty_barrier_),
          Pipeline::Stages>(sharedStorage.full_barrier_, sharedStorage.empty_barrier_, prodArvCnt,
                            consArvCnt);
    }
    // Note: fence_barrier_init() will be invoked once after all barriers are initialized.
  }

  // Consumer release the barrier.
  inline __device__ void consumer_release(PipelineState const& state) {
    mPipeline.consumer_release(state);
  }

  // Consumer test if they have to wait at the barrier.
  inline __device__ int32_t consumer_test_wait(PipelineState const& state) {
    cutlass::ConsumerToken token = mPipeline.consumer_test_wait(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Consumer try to wait at the barrier.
  inline __device__ int32_t consumer_try_wait(PipelineState const& state) {
    cutlass::ConsumerToken token = mPipeline.consumer_try_wait(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Consumer wait at the barrier.
  inline __device__ void consumer_wait(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ConsumerToken token = reinterpret_cast<cutlass::ConsumerToken const&>(t);
    mPipeline.consumer_wait(state, token);
  }

  // Producer acquire the barrier.
  inline __device__ void producer_acquire(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ProducerToken token = reinterpret_cast<cutlass::ProducerToken const&>(t);
    mPipeline.producer_acquire(state, token);
  }

  // Producer arrive at the barrier.
  inline __device__ void producer_commit(PipelineState const& state) {
    if constexpr (UseCpAsyncWaitGroup) {
      mPipeline.producer_commit(state);
    } else {
      mPipeline.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
    }
  }

  // Producer clean the cluster barrier. It does nothing.
  inline __device__ void producer_tail(PipelineState const& state) {
    mPipeline.producer_tail(state);
  }

  // Producer get the barrier. It is only used by tma load.
  inline __device__ int32_t producer_try_acquire(PipelineState const& state) {
    cutlass::ProducerToken token = mPipeline.producer_try_acquire(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Get pipeline
  [[nodiscard]] inline __device__ Pipeline& get_pipeline() { return mPipeline; }

 private:
  // The pipeline.
  Pipeline mPipeline;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassTmaAsyncPipeline class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumStages, class AtomThrShapeMNK = cute::Shape<cute::_1, cute::_1, cute::_1>>
class CutlassTmaAsyncPipeline {
  // The CUTLASS pipeline.
#if (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ != 1040)
  using Pipeline = cutlass::PipelineTmaTransformAsync<NumStages>;
#else
  using Pipeline = cutlass::PipelineTmaAsync<NumStages>;
#endif
  // The parameters to initialize the pipeline.
  using Params = typename Pipeline::Params;

 public:
  // The state.
  using PipelineState = typename Pipeline::PipelineState;
  // The shared memory storage.
  using SharedStorage = typename Pipeline::SharedStorage;
  // The producer's barrier type.
  using ProducerBarrier = typename Pipeline::ProducerBarrierType;
  // The FULL barrier type.
  using FullBarrier = typename Pipeline::FullBarrier;
  // The EMPTY barrier type.
  using EmptyBarrier = typename Pipeline::EmptyBarrier;

 public:
  // Ctor.
  template <typename ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>,
            typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  inline explicit __device__ CutlassTmaAsyncPipeline(
      SharedStorage& sharedStorage, int32_t warpId, int32_t transactionBytes = 0,
      bool isLeader = false, int32_t numConsumers = 0, ClusterShape clusterShape = {},
      InitBarriers = {}, InitMasks = {}, int32_t barInitWarpId = 0)
      : mPipeline{sharedStorage,
                  Params{reinterpret_cast<uint32_t const&>(transactionBytes),
                         Pipeline::ThreadCategory::Consumer, isLeader ? 1u : 0u,
                         reinterpret_cast<uint32_t const&>(numConsumers),
                         /* num producers */ 1u, barInitWarpId},
                  clusterShape,
                  /*InitBarriers=*/cute::false_type{}, InitMasks{}},
        mEmptyBarrierPtr{&sharedStorage.empty_barrier_[0]},
        mIsSignalingThread{threadIdx.x % numConsumers == 0} {
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      // Initialize barriers. Doing this here instead of in cutlass avoids unnecessary shuffle of
      // warp id for each pipeline and redundant predicates and branches.
      if (warpId == barInitWarpId) {
        constexpr int NumThreadsPerWarpGroup = 4 * 32;
        uint32_t const producer_arv_cnt = 1u;
        uint32_t const num_consumer_per_cluster =
            cute::ceil_div(numConsumers, static_cast<uint32_t>(NumThreadsPerWarpGroup));
#if (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ != 1040)
        auto atom_thr_shape = typename Pipeline::AtomThrShape_MNK{};
        uint32_t multicast_consumer_arrival_count =
            (cute::size<0>(clusterShape) / cute::size<0>(atom_thr_shape) +
             cute::size<1>(clusterShape) / cute::size<1>(atom_thr_shape) - 1) *
            num_consumer_per_cluster;
#else
        uint32_t multicast_consumer_arrival_count = numConsumers;  // If cluster_size is 1
        if (cute::size(clusterShape) > 1) {
          multicast_consumer_arrival_count =
              (cute::size<0>(clusterShape) + cute::size<1>(clusterShape) - 1) *
              num_consumer_per_cluster;
        }
#endif
        cutlass::arch::detail::initialize_barrier_array_pair_aligned<
            decltype(sharedStorage.full_barrier_), decltype(sharedStorage.empty_barrier_),
            Pipeline::Stages>(sharedStorage.full_barrier_, sharedStorage.empty_barrier_,
                              producer_arv_cnt, multicast_consumer_arrival_count);
      }
      // Note: fence_barrier_init() will be invoked once after all barriers are initialized.
    }
  }

  // Consumer release the barrier.
  inline __device__ void consumer_release(PipelineState const& state) {
    // NOTE: The original pipeline uses `dst_blockid_` and `signaling_thread_` to call release.
    // For CGA > 1, blockRank=X attempts to signal arrival to blockRank=0.
    // This becomes an issue in cases where .2CTA commands are not used, and no cross-CGA
    // pipelines are expected. We override this behavior by using standard `arrive` without
    // relying on the CGA blockRank or `dst_block_id`.
    if constexpr (IsMma2Sm) {
      mPipeline.consumer_release(state);
    } else {
#if (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ != 1040)
      if (mIsSignalingThread)
#endif
      {
        mEmptyBarrierPtr[state.index()].arrive();
      }
    }
  }

  // Consumer test if they have to wait at the barrier..
  inline __device__ int32_t consumer_test_wait(PipelineState const& state) {
    cutlass::ConsumerToken token = mPipeline.consumer_test_wait(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Consumer try to wait at the barrier..
  inline __device__ int32_t consumer_try_wait(PipelineState const& state) {
    cutlass::ConsumerToken token = mPipeline.consumer_try_wait(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Consumer wait at the barrier.
  inline __device__ void consumer_wait(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ConsumerToken token = reinterpret_cast<cutlass::ConsumerToken const&>(t);
    mPipeline.consumer_wait(state, token);
  }

  // Producer acquire the barrier.
  // Arrive the producer barrier with the transaction_bytes at the same time.
  inline __device__ void producer_acquire(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ProducerToken token = reinterpret_cast<cutlass::ProducerToken const&>(t);
    mPipeline.producer_acquire(state, token);
  }

  // Producer arrive at the barrier. It does nothing.
  inline __device__ void producer_commit(PipelineState const& state, uint32_t bytes = uint32_t{0}) {
    mPipeline.producer_commit(state, bytes);
  }

  // Producer clean the cluster barrier. It does nothing.
  inline __device__ void producer_tail(PipelineState const& state) {
    mPipeline.producer_tail(state);
  }

  // Producer try to acquire the barrier.
  inline __device__ int32_t producer_try_acquire(PipelineState const& state) {
    cutlass::ProducerToken token = mPipeline.producer_try_acquire(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Producer test to acquire the barrier. (Not supported by CUTLASS directly)
  inline __device__ int32_t producer_test_acquire(PipelineState const& state) {
    auto barrier = producer_get_barrier(state);
    auto phase = state.phase();
    cutlass::BarrierStatus status = cuda_ptx::mbarrier_test_wait_parity(barrier, phase)
                                        ? cutlass::BarrierStatus::WaitDone
                                        : cutlass::BarrierStatus::WaitAgain;
    cutlass::ProducerToken token = cutlass::ProducerToken(status);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Producer get the barrier. It is only used by tma load.
  inline __device__ ProducerBarrier* producer_get_barrier(PipelineState const& state) {
    return mPipeline.producer_get_barrier(state);
  }

  // Get pipeline
  [[nodiscard]] inline __device__ Pipeline& get_pipeline() { return mPipeline; }

 private:
  // The pipeline.
  Pipeline mPipeline;
  EmptyBarrier* mEmptyBarrierPtr = nullptr;
  bool mIsSignalingThread = false;
  static constexpr bool IsMma2Sm = cute::size(AtomThrShapeMNK{}) > 1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassTmaUmmaAsyncPipeline class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumStages, class ClusterShape = cute::Shape<int, int, cute::_1>,
          class AtomThrShapeMNK = cute::Shape<cute::_1, cute::_1, cute::_1>>
class CutlassTmaUmmaAsyncPipeline {
  // The CUTLASS pipeline.
  using Pipeline = cutlass::PipelineTmaUmmaAsync<NumStages, ClusterShape, AtomThrShapeMNK>;
  // The parameters to initialize the pipeline.
  using Params = typename Pipeline::Params;

 public:
  // The state.
  using PipelineState = typename Pipeline::PipelineState;
  // The shared memory storage.
  using SharedStorage = typename Pipeline::SharedStorage;
  // The producer's barrier type.
  using ProducerBarrier = typename Pipeline::ProducerBarrierType;
  // The EMPTY barrier type.
  using EmptyBarrier = typename Pipeline::EmptyBarrier;

 public:
  // Ctor.
  template <typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  inline explicit __device__ CutlassTmaUmmaAsyncPipeline(SharedStorage& sharedStorage,
                                                         int32_t warpId, uint32_t transactionBytes,
                                                         bool isLeader, ClusterShape clusterShape,
                                                         InitBarriers = {}, InitMasks = {},
                                                         int32_t barInitWarpId = 0)
      : mPipeline{sharedStorage,
                  Params{transactionBytes,
                         // Set it to producer to not create the multicast mask as the
                         // umma_peer_mask is used instead (see below).
                         Pipeline::ThreadCategory::Producer,
                         size(clusterShape) == 1 || (cute::block_id_in_cluster().x %
                                                         cute::size<0>(AtomThrShapeMNK{}) ==
                                                     0)
                             ? isLeader
                             : 0u,  // is_leader
                         0u,        // num_consumers, unused
                         0u,        // num_produces, unused
                         barInitWarpId},
                  clusterShape,
                  /*InitBarriers=*/cute::false_type{}, InitMasks{}},
        mBlockIdMask{0},
        mEmptyBarrierPtr{&sharedStorage.empty_barrier_[0]} {
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      // Initialize barriers. Doing this here instead of in cutlass avoids unnecessary shuffle of
      // warp id for each pipeline and redundant predicates and branches.
      if (warpId == barInitWarpId) {
        constexpr int producer_arv_cnt = 1;
        auto atom_thr_shape = typename Pipeline::AtomThrShape_MNK{};
        uint32_t const multicast_consumer_arrival_count =
            (cute::size<0>(clusterShape) / cute::size<0>(atom_thr_shape)) +
            (cute::size<1>(clusterShape) / cute::size<1>(atom_thr_shape)) - 1;
        cutlass::arch::detail::initialize_barrier_array_pair_aligned<
            decltype(sharedStorage.full_barrier_), decltype(sharedStorage.empty_barrier_),
            Pipeline::Stages>(sharedStorage.full_barrier_, sharedStorage.empty_barrier_,
                              producer_arv_cnt, multicast_consumer_arrival_count);
      }
      // Note: fence_barrier_init() will be invoked once after all barriers are initialized.
    }

    // Initialize the blockId mask.
    init_block_id_mask(clusterShape);
  }

  // Create the umma_peer_mask, which only encodes the current CTA pair in the cluster.
  // This overrides the base implementation of init_block_id_mask, which creates the multicast
  // mask that encodes all CTA pairs in the cluster.
  __device__ void init_block_id_mask(ClusterShape clusterShape,
                                     dim3 blockIdInCluster = cute::block_id_in_cluster()) {
    if constexpr (IsMma2Sm) {
      auto cluster_layout = cute::make_layout(clusterShape);
      mBlockIdMask = cutlass::detail::calculate_umma_peer_mask(clusterShape, AtomThrShapeMNK{},
                                                               blockIdInCluster);
    }
  }

  // Consumer release the barrier.
  inline __device__ void consumer_release(PipelineState const& state) {
    // Override the original consumer_release to use the umma_peer_mask instead of multicast mask.
    uint64_t* smemPtr = reinterpret_cast<uint64_t*>(&mEmptyBarrierPtr[state.index()]);
    if constexpr (IsMma2Sm) {
      cutlass::arch::umma_arrive_multicast_2x1SM(smemPtr, mBlockIdMask);
    } else {
      cutlass::arch::umma_arrive(smemPtr);
    }
  }

  // Consumer try to wait at the barrier..
  inline __device__ int32_t consumer_try_wait(PipelineState const& state) {
    cutlass::ConsumerToken token = mPipeline.consumer_try_wait(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Consumer wait at the barrier.
  inline __device__ void consumer_wait(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ConsumerToken token = reinterpret_cast<cutlass::ConsumerToken const&>(t);
    mPipeline.consumer_wait(state, token);
  }

  // Producer acquire the barrier.
  // Arrive the producer barrier with the transaction_bytes at the same time.
  inline __device__ void producer_acquire(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ProducerToken token = reinterpret_cast<cutlass::ProducerToken const&>(t);
    mPipeline.producer_acquire(state, token);
  }

  // Producer arrive at the barrier. It does nothing.
  inline __device__ void producer_commit(PipelineState const& state, uint32_t bytes = uint32_t{0}) {
    mPipeline.producer_commit(state, bytes);
  }

  // Producer clean the cluster barrier. It does nothing.
  inline __device__ void producer_tail(PipelineState const& state) {
    mPipeline.producer_tail(state);
  }

  // Producer try to acquire the barrier.
  inline __device__ int32_t producer_try_acquire(PipelineState const& state) {
    cutlass::ProducerToken token = mPipeline.producer_try_acquire(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Producer get the barrier. It is only used by tma load.
  inline __device__ ProducerBarrier* producer_get_barrier(PipelineState const& state) {
    return mPipeline.producer_get_barrier(state);
  }

  // Get pipeline
  [[nodiscard]] inline __device__ Pipeline& get_pipeline() { return mPipeline; }

 private:
  // The pipeline.
  Pipeline mPipeline;
  // The blockId mask.
  uint16_t mBlockIdMask;
  // The empty barrier pointer.
  EmptyBarrier* mEmptyBarrierPtr;
  // Does it use 2CTA UTCMMA ?
  static constexpr bool IsMma2Sm = cute::size(AtomThrShapeMNK{}) > 1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumStages, cutlass::McastDirection mcastDirection = cutlass::McastDirection::kRowCol,
          class ClusterShape = cute::Shape<int, int, cute::_1>,
          class AtomThrShapeMNK = cute::Shape<cute::_1, cute::_1, cute::_1>>
class CutlassTmaMultiUmmaAsyncPipeline {
  // The CUTLASS pipeline.
  using Pipeline =
      cutlass::PipelineTmaMultiUmmaAsync<NumStages, mcastDirection, ClusterShape, AtomThrShapeMNK>;
  // The parameters to initialize the pipeline.
  using Params = typename Pipeline::Params;

 public:
  // The state.
  using PipelineState = typename Pipeline::PipelineState;
  // The shared memory storage.
  using SharedStorage = typename Pipeline::SharedStorage;
  // The producer's barrier type.
  using ProducerBarrier = typename Pipeline::ProducerBarrierType;

 public:
  // Ctor.
  template <typename InitBarriers = cute::true_type, typename InitMasks = cute::true_type>
  inline explicit __device__ CutlassTmaMultiUmmaAsyncPipeline(
      SharedStorage& sharedStorage, int32_t warpId, uint32_t transactionBytes, bool isLeader,
      uint32_t numConsumers, ClusterShape clusterShape, InitBarriers = {}, InitMasks = {},
      int32_t barInitWarpId = 0)
      : mPipeline{
            sharedStorage,
            warpId,
            Params{reinterpret_cast<uint32_t const&>(transactionBytes),
                   Pipeline::ThreadCategory::Consumer,
                   // just use multiple CTAs in the X dimension of the cluster.
                   size(clusterShape) == 1 ||
                           (cute::block_id_in_cluster().x % cute::size<0>(AtomThrShapeMNK{}) == 0)
                       ? isLeader
                       : 0u,      // is_leader
                   numConsumers,  // num_consumers
                   0u,            // num_producers, unused
                   barInitWarpId},
            clusterShape,
            InitBarriers{},
            InitMasks{}} {}

  // Consumer release the barrier.
  inline __device__ void consumer_release(PipelineState const& state) {
    mPipeline.consumer_release(state);
  }

  // Consumer try to wait at the barrier..
  inline __device__ int32_t consumer_try_wait(PipelineState const& state) {
    cutlass::ConsumerToken token = mPipeline.consumer_try_wait(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Consumer wait at the barrier.
  inline __device__ void consumer_wait(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ConsumerToken token = reinterpret_cast<cutlass::ConsumerToken const&>(t);
    mPipeline.consumer_wait(state, token);
  }

  // Producer acquire the barrier.
  // Arrive the producer barrier with the transaction_bytes at the same time.
  inline __device__ void producer_acquire(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ProducerToken token = reinterpret_cast<cutlass::ProducerToken const&>(t);
    mPipeline.producer_acquire(state, token);
  }

  // Producer arrive at the barrier. It does nothing.
  inline __device__ void producer_commit(PipelineState const& state, uint32_t bytes = uint32_t{0}) {
    mPipeline.producer_commit(state, bytes);
  }

  // Producer clean the cluster barrier. It does nothing.
  inline __device__ void producer_tail(PipelineState const& state) {
    mPipeline.producer_tail(state);
  }

  // Producer try to acquire the barrier.
  inline __device__ int32_t producer_try_acquire(PipelineState const& state) {
    cutlass::ProducerToken token = mPipeline.producer_try_acquire(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Producer get the barrier. It is only used by tma load.
  inline __device__ ProducerBarrier* producer_get_barrier(PipelineState const& state) {
    return mPipeline.producer_get_barrier(state);
  }

  // Get pipeline
  [[nodiscard]] inline __device__ Pipeline& get_pipeline() { return mPipeline; }

 private:
  // The pipeline.
  Pipeline mPipeline;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassUmmaAsyncPipeline class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumStages, class AtomThrShapeMNK = cute::Shape<cute::_1, cute::_1, cute::_1>>
class CutlassUmmaAsyncPipeline {
  // The CUTLASS pipeline.
  using Pipeline = cutlass::PipelineUmmaAsync<NumStages, AtomThrShapeMNK>;
  // The parameters to initialize the pipeline.
  using Params = typename Pipeline::Params;

 public:
  // The state.
  using PipelineState = typename Pipeline::PipelineState;
  // The shared memory storage.
  using SharedStorage = typename Pipeline::SharedStorage;
  // The producer's barrier type.
  using ProducerBarrier = typename Pipeline::ProducerBarrierType;
  // The EMPTY barrier type.
  using EmptyBarrier = typename Pipeline::EmptyBarrier;

 public:
  // Ctor.
  template <typename ClusterShape, typename InitBarriers = cute::true_type,
            typename InitMasks = cute::true_type>
  inline explicit __device__ CutlassUmmaAsyncPipeline(SharedStorage& sharedStorage, int32_t warpId,
                                                      int32_t consArvCnt, ClusterShape clusterShape,
                                                      InitBarriers = {}, InitMasks = {},
                                                      int32_t barInitWarpId = 0)
      : mPipeline{sharedStorage,
                  Params{Pipeline::ThreadCategory::Producer,
                         1u,                                             // prod_cnt
                         reinterpret_cast<uint32_t const&>(consArvCnt),  // cons_cnt
                         cute::block_rank_in_cluster(),                  // dst block id
                         barInitWarpId},
                  clusterShape,
                  /*InitBarriers=*/cute::false_type{}, InitMasks{}},
        mEmptyBarrierPtr{&sharedStorage.empty_barrier_[0]} {
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      // Initialize barriers. Doing this here instead of in cutlass avoids unnecessary shuffle of
      // warp id for each pipeline and redundant predicates and branches.
      if (warpId == barInitWarpId) {
        cutlass::arch::detail::initialize_barrier_array_pair_aligned<
            decltype(sharedStorage.full_barrier_), decltype(sharedStorage.empty_barrier_),
            Pipeline::Stages>(sharedStorage.full_barrier_, sharedStorage.empty_barrier_, 1u,
                              consArvCnt);
      }
    }
    // Note: fence_barrier_init() will be invoked once after all barriers are initialized.
  }

  // Consumer release the barrier.
  inline __device__ void consumer_release(PipelineState const& state) {
    // NOTE: The original pipeline uses `dst_blockid_` and `signaling_thread_` to call release.
    // For CGA > 1, blockRank=X attempts to signal arrival to blockRank=0.
    // This becomes an issue in cases where .2CTA commands are not used, and no cross-CGA
    // pipelines are expected. We override this behavior by using standard `arrive` without
    // relying on the CGA blockRank or `dst_block_id`.
    if constexpr (IsMma2Sm) {
      mPipeline.consumer_release(state);
    } else {
      mEmptyBarrierPtr[state.index()].arrive();
    }
  }

  // Consumer try to wait at the barrier..
  inline __device__ int32_t consumer_try_wait(PipelineState const& state) {
    cutlass::ConsumerToken token = mPipeline.consumer_try_wait(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Consumer wait at the barrier.
  inline __device__ void consumer_wait(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ConsumerToken token = reinterpret_cast<cutlass::ConsumerToken const&>(t);
    mPipeline.consumer_wait(state, token);
  }

  // Producer acquire the barrier.
  inline __device__ void producer_acquire(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ProducerToken token = reinterpret_cast<cutlass::ProducerToken const&>(t);
    mPipeline.producer_acquire(state, token);
  }

  // Producer arrive at the barrier. It does nothing.
  inline __device__ void producer_commit(PipelineState const& state) {
    mPipeline.producer_commit(state);
  }

  // Producer clean the cluster barrier. It does nothing.
  inline __device__ void producer_tail(PipelineState const& state) {
    mPipeline.producer_tail(state);
  }

  // Producer try to acquire the barrier.
  inline __device__ int32_t producer_try_acquire(PipelineState const& state) {
    cutlass::ProducerToken token = mPipeline.producer_try_acquire(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Get pipeline
  [[nodiscard]] inline __device__ Pipeline& get_pipeline() { return mPipeline; }

 private:
  // The pipeline.
  Pipeline mPipeline;
  EmptyBarrier* mEmptyBarrierPtr = nullptr;
  static constexpr bool IsMma2Sm = cute::size(AtomThrShapeMNK{}) > 1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassUmmaConsumerAsyncPipeline class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumStages, bool UsesCpAsyncBarrierArrive = false,
          bool UsesFenceBeforeProdCommit = false,
          class AtomThrShapeMNK = cute::Shape<cute::_1, cute::_1, cute::_1>>
class CutlassUmmaConsumerAsyncPipeline {
  // The CUTLASS pipeline.
  using Pipeline = cutlass::PipelineUmmaConsumerAsync<NumStages, AtomThrShapeMNK>;
  // The parameters to initialize the pipeline.
  using Params = typename Pipeline::Params;

 public:
  // The state.
  using PipelineState = typename Pipeline::PipelineState;
  // The shared memory storage.
  using SharedStorage = typename Pipeline::SharedStorage;
  // The producer's barrier type.
  using ProducerBarrier = typename Pipeline::ProducerBarrierType;

 public:
  // Ctor.
  template <typename ClusterShape, typename InitBarriers = cute::true_type,
            typename InitMasks = cute::true_type>
  inline explicit __device__ CutlassUmmaConsumerAsyncPipeline(SharedStorage& sharedStorage,
                                                              int32_t warpId, int32_t prodCnt,
                                                              ClusterShape clusterShape,
                                                              InitBarriers = {}, InitMasks = {},
                                                              int32_t barInitWarpId = 0)
      : mPipeline{sharedStorage,
                  Params{Pipeline::ThreadCategory::Consumer,
                         reinterpret_cast<uint32_t const&>(prodCnt),  // prod_cnt
                         1u,                                          // cons_cnt
                         0u,                                          // dst block id
                         barInitWarpId},
                  clusterShape,
                  /*InitBarriers=*/cute::false_type{}, InitMasks{}} {
    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      // Initialize barriers. Doing this here instead of in cutlass avoids unnecessary shuffle of
      // warp id for each pipeline and redundant predicates and branches.
      if (warpId == barInitWarpId) {
        cutlass::arch::detail::initialize_barrier_array_pair_aligned<
            decltype(sharedStorage.full_barrier_), decltype(sharedStorage.empty_barrier_),
            Pipeline::Stages>(sharedStorage.full_barrier_, sharedStorage.empty_barrier_, prodCnt,
                              1u);
      }
    }
    // Note: fence_barrier_init() will be invoked once after all barriers are initialized.
  }

  // Consumer release the barrier.
  inline __device__ void consumer_release(PipelineState const& state) {
    mPipeline.consumer_release(state);
  }

  // Consumer try to wait at the barrier..
  inline __device__ int32_t consumer_try_wait(PipelineState const& state) {
    cutlass::ConsumerToken token = mPipeline.consumer_try_wait(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Consumer wait at the barrier.
  inline __device__ void consumer_wait(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ConsumerToken token = reinterpret_cast<cutlass::ConsumerToken const&>(t);
    mPipeline.consumer_wait(state, token);
  }

  // Producer acquire the barrier.
  inline __device__ void producer_acquire(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ProducerToken token = reinterpret_cast<cutlass::ProducerToken const&>(t);
    mPipeline.producer_acquire(state, token);
  }

  // Producer arrive at the barrier. It does nothing.
  inline __device__ void producer_commit(PipelineState const& state) {
    if constexpr (UsesCpAsyncBarrierArrive) {
      mPipeline.producer_commit(state, cutlass::arch::cpasync_barrier_arrive);
    } else {
      if constexpr (UsesFenceBeforeProdCommit) {
        // Need a fence when we need to make the local STS visible to the other CTA in the CGA
        asm volatile(
            "{\n\t"
            "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;\n\t"
            "}");
      }
      mPipeline.producer_commit(state);
    }
  }

  // Producer clean the cluster barrier. It does nothing.
  inline __device__ void producer_tail(PipelineState const& state) {
    mPipeline.producer_tail(state);
  }

  // Producer try to acquire the barrier.
  inline __device__ int32_t producer_try_acquire(PipelineState const& state) {
    cutlass::ProducerToken token = mPipeline.producer_try_acquire(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Get pipeline
  [[nodiscard]] inline __device__ Pipeline& get_pipeline() { return mPipeline; }

 private:
  // The pipeline.
  Pipeline mPipeline;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// ClcFastDrain class
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t DrainRate_ = 4>
class ClcFastDrain {
 public:
  static constexpr int32_t DrainRate = DrainRate_;

  // Shared memory storage.
  struct SharedStorage {
    alignas(16) uint4 responses[DrainRate];  // 16 bytes per CLC response
    alignas(8) uint64_t mbar;                // mbarrier for async completion
  };

  // Constructor - initializes mbarrier
  inline __device__ ClcFastDrain(SharedStorage& storage)
      : mResponsesPtr(storage.responses), mMbarPtr(&storage.mbar), mParity(0) {
    // Only one thread should initialize the mbarrier.
    if (cute::elect_one_sync()) {
      // Initialize mbarrier with arrival count of 1.
      cuda_ptx::mbarrier_init(mMbarPtr, 1u);
      cuda_ptx::fence_mbarrier_init(cuda_ptx::sem_release, cuda_ptx::scope_cluster);
    }
  }

  // Issue one batch of DrainRate CLC cancel queries.
  inline __device__ int32_t drain_batch() {
    int32_t const parity = mParity;

    // Only one thread issues commands
    if (cute::elect_one_sync()) {
      // Arrive at mbarrier and set expected transaction bytes.
      cuda_ptx::mbarrier_arrive_expect_tx(cuda_ptx::sem_relaxed, cuda_ptx::scope_cta,
                                          cuda_ptx::space_shared, mMbarPtr,
                                          static_cast<uint32_t>(sizeof(uint4) * DrainRate));

      // Issue several back-to-back CLC try_cancel commands.
#pragma unroll
      for (int32_t i = 0; i < DrainRate; ++i) {
        cuda_ptx::clusterlaunchcontrol_try_cancel(&mResponsesPtr[i], mMbarPtr);
      }
    }

    // All threads wait for responses to arrive.
    while (!cuda_ptx::mbarrier_try_wait_parity(mMbarPtr, static_cast<uint32_t>(parity))) {
    }

    // Count successful cancellations.
    int32_t canceled_count = 0;
#pragma unroll
    for (int32_t i = 0; i < DrainRate; ++i) {
      uint4 const response = mResponsesPtr[i];
      if (cuda_ptx::clusterlaunchcontrol_query_cancel_is_canceled(response)) {
        ++canceled_count;
      }
    }

    // Flip parity for next iteration.
    mParity ^= 1;

    return canceled_count;
  }

  // Drain all pending CTAs until queue is empty.
  inline __device__ int32_t drain_all() {
    int32_t total_drained = 0;
    int32_t drained;

    do {
      drained = drain_batch();
      total_drained += drained;
    } while (drained > 0);

    return total_drained;
  }

 private:
  uint4* mResponsesPtr;
  uint64_t* mMbarPtr;
  int32_t mParity;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// CutlassWorkIdPipeline class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumStages, class ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>>
class CutlassWorkIdPipeline {
  // The CUTLASS pipeline.
  using Pipeline = cutlass::PipelineCLCFetchAsync<NumStages, ClusterShape>;
  // The parameters to initialize the pipeline.
  using Params = typename Pipeline::Params;

 public:
  // The state.
  using PipelineState = typename Pipeline::PipelineState;
  // The shared memory storage.
  using SharedStorage = typename Pipeline::SharedStorage;

 public:
  // Ctor.
  inline explicit __device__ CutlassWorkIdPipeline(SharedStorage& sharedStorage,
                                                   ClusterShape clusterShape = ClusterShape{},
                                                   int32_t prodArvCnt = 1, int32_t consArvCnt = 1,
                                                   int32_t prodBlkId = 0, int32_t barInitWarpId = 0)
      : mPipeline{sharedStorage,
                  Params{/*transaction_bytes=*/16,
                         /*role=*/Pipeline::ThreadCategory::NonParticipant,
                         /*is_leader=*/0,
                         /*num_consumers=*/0,
                         /*producer_blockid=*/reinterpret_cast<uint32_t const&>(prodBlkId),
                         /*producer_arv_count=*/reinterpret_cast<uint32_t const&>(prodArvCnt),
                         /*consumer_arv_count=*/reinterpret_cast<uint32_t const&>(consArvCnt),
                         barInitWarpId},
                  clusterShape} {}

  // Consumer release the barrier.
  inline __device__ void consumer_release(PipelineState const& state) {
    mPipeline.consumer_release(state);
  }

  // Consumer try to wait at the barrier..
  [[nodiscard]] inline __device__ int32_t consumer_try_wait(PipelineState const& state) {
    cutlass::ConsumerToken token = mPipeline.consumer_try_wait(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Consumer wait at the barrier.
  inline __device__ void consumer_wait(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ConsumerToken token = reinterpret_cast<cutlass::ConsumerToken const&>(t);
    mPipeline.consumer_wait(state, token);
  }

  // Producer acquire the barrier.
  inline __device__ void producer_acquire(PipelineState const& state, int32_t t = int32_t{0}) {
    cutlass::ProducerToken token = reinterpret_cast<cutlass::ProducerToken const&>(t);
    mPipeline.producer_acquire(state, token);
  }

  // Producer clean the cluster barrier. It does nothing.
  inline __device__ void producer_tail(PipelineState const& state) {
    mPipeline.producer_tail(state);
  }

  // Producer get the barrier. It is only used by tma load.
  [[nodiscard]] inline __device__ int32_t producer_try_acquire(PipelineState const& state) {
    cutlass::ProducerToken token = mPipeline.producer_try_acquire(state);
    return reinterpret_cast<int32_t const&>(token);
  }

  // Get producer mbarrier address
  [[nodiscard]] inline __device__ uint32_t producer_get_barrier(PipelineState const& state) {
    return mPipeline.producer_get_barrier(state);
  }

  // Get pipeline
  [[nodiscard]] inline __device__ Pipeline& get_pipeline() { return mPipeline; }

  // Fast drain rate for CLC cancel operations
  static constexpr int32_t FastDrainRate = 4;

  // Fast drain shared memory storage type
  using FastDrainStorage = typename ClcFastDrain<FastDrainRate>::SharedStorage;

  // Fast drain all pending CTAs using ClcFastDrain.
  // This accelerates draining CTAs in queue instead of draining one-by-one, if we know we can have
  // a clean break from the loop when, for example, rastering CTA in the non-batch dimension.
  inline __device__ int32_t fast_drain_all(FastDrainStorage& fastDrainStorage) {
    ClcFastDrain<FastDrainRate> drainer(fastDrainStorage);
    return drainer.drain_all();
  }

 private:
  // The pipeline.
  Pipeline mPipeline;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// PipelineNonBlockingState class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <uint32_t Stages_, bool SequentialAcquire_>
struct PipelineNonBlockingState {
  // Number of stages in pipeline.
  static constexpr int32_t Stages = Stages_;

  // Use sequential offset tracking.
  static constexpr bool SequentialAcquire = SequentialAcquire_;

  // Mask of stages being used.
  static constexpr uint32_t StageMask = (0b1 << Stages) - 1;

  // If the stages of this pipeline are being consumed in a sequential order, the offset of each
  // stage can be implicitly tracked with a global offset. This approach avoids eliminates register
  // array indexing, so can perform better when the DMA loop is not unrolled. However, if the
  // non-blocking loop can be unrolled Stages times, it is faster to explicitly store the offsets
  // (for Gemm at least).
  //
  // For a 3 stage pipeline, the tiles will be mapped to the different stages as follows with stage
  // 0 receiving tiles {0, 3, 6, ...}, stage 1 receiving tile {1, 4, 7, ...}, and stage 2 receiving
  // tiles {2, 5, 8, ...}.
  //
  // The pipeline iteration groups together one tile for each stage in the pipeline, so pipeline
  // iteration 0 corresponds to tiles 0, 1, and 2. Each pipeline iteration will have a corresponding
  // global offset (pipeline iter * number of stages), and the offset for each stage can be computed
  // relative to this.
  //
  // =========================================
  //               Tile ordering
  // =========================================
  // ┌───────────────┬───┬───┬───┬───┬───┬───┐
  // │ Pipeline Iter │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
  // │ Global offset │ 0 │ 3 │ 6 │ 9 │12 │15 │
  // │ Global phase  │ 0 │ 1 │ 0 │ 1 │ 0 │ 1 │
  // ├───────────────┼───┼───┼───┼───┼───┼───┤
  // │     Stage   0 │ 0 │ 3 │ 6 │ 9 │   │   │
  // │     index   1 │ 1 │ 4 │ 7 │...│   │   │
  // │             2 │ 2 │ 5 │ 8 │   │   │   │
  // └───────────────┴───┴───┴───┴───┴───┴───┘
  //
  // =========================================
  //          Potential pipeline state
  // =========================================
  // The sequential tracking will keep track of mGlobalOffset and mGlobalPhase as seen above.
  // Additionally, the phase of the currently acquired tile will be stored for each tile. This is
  // different from the conditional phase which gets flipped when a stage is committed. The offset
  // for a given stage can then be computed via:
  //
  //   offset = mGlobalOffset + numStage * (acquirePhase == globalPhase) + stageIndex
  //
  // Finally, a running count of the number of acquires which have been acquired during the current
  // pipeline iteration will be stored. Once this reaches the total number of stages, the global
  // state will be updated.
  //
  // =========================================
  //          Potential pipeline state
  // =========================================
  // ┌───────────────┬───┬───┬───┬───┬───┬───┐
  // │ Pipeline Iter │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
  // │ Phase         │ 0 │ 1 │ 0 │ 1 │ 0 │ 1 │
  // ├───────────────┼───┼───┼───┼───┼───┼───┤
  // │     Stage   0 │ • │ • │ A │   │   │   │  A = Acquired
  // │     index   1 │ • │ • │ A │   │   │   │  • = Complete
  // │             2 │ • │ A │   │   │   │   │
  // └───────────────┴───┴───┴───┴───┴───┴───┘
  //                       ↑
  //                   Global state (mGlobalOffset = 3, mGlobalPhase = 1, mAcquireCount = 2)
  //
  //  Stage 0 offset: mGlobalOffset (3) + non-matching phases (3) + stage index (0) = 6
  //  Stage 1 offset: mGlobalOffset (3) + non-matching phases (3) + stage index (1) = 7
  //  Stage 2 offset: mGlobalOffset (3) +     matching phases (0) + stage index (2) = 5
  //
  // =========================================
  //            Stage 2 acquires
  // =========================================
  // Here stage 2 successfully finishes work on tile 5 and then successfully acquires tile 8. Since
  // the acquire count would reach the number of stage (3), the global offset and global phase would
  // be updated. Since stage 0 and stage 1 now have phases which match the global phase, their
  // offsets remain unchanged.
  //
  // ┌───────────────┬───┬───┬───┬───┬───┬───┐
  // │ Pipeline Iter │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
  // │ Phase         │ 0 │ 1 │ 0 │ 1 │ 0 │ 1 │
  // ├───────────────┼───┼───┼───┼───┼───┼───┤
  // │     Stage   0 │ • │ • │ A │   │   │   │  A = Acquired
  // │     index   1 │ • │ • │ A │   │   │   │  • = Complete
  // │             2 │ • │ • │ A │   │   │   │
  // └───────────────┴───┴───┴───┴───┴───┴───┘
  //                           ↑
  //                       Global state (mGlobalOffset = 6, mGlobalPhase = 0, mAcquireCount = 0)
  //  Stage 0 offset: mGlobalOffset (6) +     matching phases (0) + stage index (0) = 6
  //  Stage 1 offset: mGlobalOffset (6) +     matching phases (0) + stage index (1) = 7
  //  Stage 2 offset: mGlobalOffset (6) +     matching phases (0) + stage index (2) = 8
  //
  //
  // =========================================
  //           Stage 0 commits
  // =========================================
  //  If stage 0 finishes work on tile 6, it will not be able to successfully acquire tile 9 as the
  //  consumer is processing tiles in sequential order and the consumer has not been able to start
  //  work on tile 5 yet. The explicit offsets are currently updated in commitAndIncrement function,
  //  but for the sequential tracking, the update logic needs to happen in the acquire logic.
  //
  // ┌───────────────┬───┬───┬───┬───┬───┬───┐
  // │ Pipeline Iter │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
  // │ Phase         │ 0 │ 1 │ 0 │ 1 │ 0 │ 1 │
  // ├───────────────┼───┼───┼───┼───┼───┼───┤
  // │     Stage   0 │ • │ • │ • │   │   │   │  A = Acquired
  // │     index   1 │ • │ • │ A │   │   │   │  • = Complete
  // │             2 │ • │ A │   │   │   │   │
  // └───────────────┴───┴───┴───┴───┴───┴───┘
  //                       ↑
  //                   Global state (mGlobalOffset = 3, mGlobalPhase = 1, mAcquireCount = 2)
  //
  //  Stage 0 offset:
  //  Stage 1 offset: mGlobalOffset (3) + non-matching phases (3) + stage index (1) = 7
  //  Stage 2 offset: mGlobalOffset (3) +     matching phases (0) + stage index (2) = 6

  // =========================================
  //    Stage 0,2 commit, Stage 0 acquires
  // =========================================
  // Once stage 2 has committed tile 5, the consumer task can continue working, and there is no
  // guarantee that tile 8 will be acquired prior to tile 9; however, either way, the acquire count
  // will reach the number of stages and trigger a global state update. Since stage 2 has not
  // acquired its next tile yet, the offset function would technically return an incorrect value,
  // but since tile 5 has been completed, the wrong value will never be computed. Once stage 2
  // acquires tile 8, its acquire phase will be updated to match the global phase and the offset
  // calculation will be correct.
  //
  // ┌───────────────┬───┬───┬───┬───┬───┬───┐
  // │ Pipeline Iter │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │
  // │ Phase         │ 0 │ 1 │ 0 │ 1 │ 0 │ 1 │
  // ├───────────────┼───┼───┼───┼───┼───┼───┤
  // │     Stage   0 │ • │ • │ • │ A │   │   │  A = Acquired
  // │     index   1 │ • │ • │ A │   │   │   │  • = Complete
  // │             2 │ • │ • │   │   │   │   │
  // └───────────────┴───┴───┴───┴───┴───┴───┘
  //                           ↑
  //                       Global state (mGlobalOffset = 6, mGlobalPhase = 0, mAcquireCount = 0)
  //  Stage 0 offset: mGlobalOffset (6) + non-matching phases (3) + stage index (0) = 9
  //  Stage 1 offset: mGlobalOffset (6) +     matching phases (0) + stage index (1) = 7
  //  Stage 2 offset:

  // Use of 32-bit wide bit-vectors limits number of stages.
  static_assert(Stages <= 32, "Number of stages for non-blocking pipeline must be less than 32");

  // Number of acquired stages during current phase. (Sequential tracking only).
  int32_t mAcquiredCount{0};
  // Bit-vector with acquired phase. (Sequential tracking only).
  uint32_t mAcquiredPhases{0};
  // Bit-vector with conditional phases for each stage.
  uint32_t mConditionalPhases{0};
  // Global offset. (Sequential tracking only).
  int32_t mGlobalOffset{0};
  // Global phase. (Sequential tracking only).
  uint32_t mGlobalPhase{1};
  // Bit-vector with isAcquired state for each stage.
  uint32_t mIsAcquired{0};
  // Ending value for loop iteration.
  int32_t mLoopEnd{0};
  // Number of loop iterations.
  int32_t mLoopIterations{0};
  // Explicitly store the offset for each stage. (Explicit tracking only).
  int32_t mLoopOffset[Stages];
  // Starting value for loop iteration.
  int32_t mLoopStart{0};
  // Step size for loop.
  int32_t mLoopStep{0};
  // Bit-vector with primary phases for each stage.
  uint32_t mPrimaryPhases{0};
  // Step size for overall pipeline.
  int32_t const mPipelineStep{0};
  // Bit-vector with stageComplete state for each stage.
  uint32_t mStageComplete{0};
  // Stage offset for this workId (used for persistent kernels)
  int32_t mWorkIdOffset{0};

  __device__ inline PipelineNonBlockingState(int32_t const loopStart, int32_t const loopEnd,
                                             int32_t const loopStep)
      : mConditionalPhases(StageMask),
        mIsAcquired(0),
        mLoopEnd(loopEnd),
        mLoopIterations((loopEnd - loopStart) / loopStep),
        mLoopStart(loopStart),
        mLoopStep(loopStep),
        mPrimaryPhases(StageMask),
        mPipelineStep{Stages * mLoopStep},
        mStageComplete(0),
        mWorkIdOffset(0) {
    if constexpr (SequentialAcquire) {
      mGlobalOffset = mLoopStart - mPipelineStep;
      mGlobalPhase = 1;
      mAcquiredPhases = StageMask;
      mAcquiredCount = 0;
#pragma unroll
      for (uint32_t ii = 0; ii < Stages; ++ii) {
        updateStageComplete(mLoopStart + mLoopStep * ii, ii);
      }
    } else {
#pragma unroll
      for (uint32_t ii = 0; ii < Stages; ++ii) {
        mLoopOffset[ii] = mLoopStart + mLoopStep * ii;
        updateStageComplete(offset(ii), ii);
      }
    }
  }

  // Returns if all stages have finished work.
  __device__ inline uint32_t allStageComplete() const { return (mStageComplete == StageMask); }

  // Updates the current state after a stage has been committed
  __device__ inline void commitAndIncrement(int32_t const index) {
    // When SequentialAcquire is being used, the offset is only updated once the next tile has
    // been acquired, so the stageComplete bit-vector should be updated by adding the pipelineStep
    // to the current offset
    updateStageComplete(offset(index) + mPipelineStep, index);

    mConditionalPhases ^= (0b1 << index);
    mIsAcquired &= ~(0b1 << index);
    if constexpr (!SequentialAcquire) {
      mLoopOffset[index] += mPipelineStep;
    }
  }

  // Returns the conditional phase for the given stage.
  __device__ inline uint32_t conditionalPhase(int32_t const index) const {
    return (mConditionalPhases >> index) & 0b1;
  }

  // Flips the primary phase.
  __device__ inline void flipPrimary(int32_t const index) { mPrimaryPhases ^= (0b1 << index); }

  // Returns the next stage index.
  __device__ inline int32_t getNextStageIndex(int32_t index) const {
    index++;
    if (index == Stages) {
      index = 0;
    }
    return index;
  }

  // Returns if this is the first iteration for this stage.
  __device__ inline int32_t isFirstIter(int32_t const index) const {
    if constexpr (SequentialAcquire) {
      return mGlobalOffset == (mLoopStart - mPipelineStep);
    } else {
      return offset(index) == index;
    }
  }

  // Returns if the current stage is acquired.
  __device__ inline uint32_t isAcquired(int32_t const index) const {
    return (mIsAcquired >> index) & 0b1;
  }

  // Returns if a stage has completed all its tiles.
  __device__ inline uint32_t isStageComplete(int32_t const index) const {
    return (mStageComplete >> index) & 0b1;
  }

  // Returns the offset for the given stage.
  __device__ inline int32_t offset(int32_t const index) const {
    if constexpr (SequentialAcquire) {
      int32_t const offsetIndex = (index + Stages - mWorkIdOffset) % Stages;
      int32_t const step =
          offsetIndex + Stages * (mGlobalPhase != ((mAcquiredPhases >> index) & 0b1));
      return mGlobalOffset + step * mLoopStep;
    } else {
      return mLoopOffset[index];
    }
  }

  // Returns the primary phase for the given stage.
  __device__ inline uint32_t primaryPhase(int32_t const index) const {
    return (mPrimaryPhases >> index) & 0b1;
  }

  // Resets the barrier for a new workId and returns the starting stageIdx (used in persistent
  // scheduled kernels).
  //
  // For example for a persistent scheduler, if we have 3 stages, tileK=256, and K=512, each CTA
  // would have 2 k tiles. The assignment of these tiles to the stages is shown in the table below.
  // When moving to the next work Id, the stage index does not completely reset, it just moves to
  // the next one in order.
  //
  // ┌───────────────┬───────────────────────┬───────────────────────┐
  // │ Pipeline Iter │          0            │          1            │
  // ├───────────────┼───────────────────────┼───────────────────────┤
  // │     Stage   0 │ workId:0 k tile idx:0 │ workId:1 k tile idx:1 │
  // │     index   1 │ workId:0 k tile idx:1 │ workId:2 k tile idx:0 │
  // │             2 │ workId:1 k tile idx:0 │ workId:2 k tile idx:1 │
  // └───────────────┴───────────────────────┴───────────────────────┘
  __device__ inline int32_t resetForNextWorkId() {
    // Compute the new stage offset for this workId
    mWorkIdOffset = (mWorkIdOffset + mLoopIterations) % Stages;
    // Reset bit-vectors to indicate that none of the stages are acquired or complete
    mIsAcquired = 0u;
    mStageComplete = 0u;
    // Update the tile offset data
    if constexpr (SequentialAcquire) {
      mGlobalOffset = mLoopStart - mPipelineStep;
      mGlobalPhase = 1;
      mAcquiredPhases = StageMask;
      mAcquiredCount = 0;
#pragma unroll
      for (uint32_t ii = 0; ii < Stages; ++ii) {
        int32_t const startingIndex = (ii + Stages - mWorkIdOffset) % Stages;
        updateStageComplete(mLoopStart + mLoopStep * startingIndex, ii);
      }
    } else {
#pragma unroll
      for (int32_t ii = 0; ii < Stages; ++ii) {
        int32_t const startingIndex = (ii + Stages - mWorkIdOffset) % Stages;
        mLoopOffset[ii] = mLoopStart + mLoopStep * startingIndex;
        updateStageComplete(offset(ii), ii);
      }
    }

    return mWorkIdOffset;
  }

  // Indicates that the current stage has successfully been acquired
  __device__ inline void setAcquired(int32_t const index) {
    mIsAcquired |= (0b1 << index);

    if constexpr (SequentialAcquire) {
      mAcquiredPhases ^= (0b1 << index);
      mAcquiredCount++;
      if (mAcquiredCount == Stages) {
        mAcquiredCount = 0;
        mGlobalOffset += mPipelineStep;
        mGlobalPhase ^= 1;
      }
    }
  }

  // Checks if a tile has finish all its tiles and updates the state
  __device__ inline void updateStageComplete(int32_t const stageOffset, int32_t const index) {
    uint32_t isDone = (stageOffset >= mLoopEnd) ? 1 : 0;
    mStageComplete |= (isDone << index);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// PipelineLamport class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages_, bool UseUmma_>
class PipelineLamport {
 public:
  // Number of stages in pipeline.
  static constexpr uint32_t Stages = Stages_;
  // Is the consumer use umma.
  static constexpr bool UseUmma = UseUmma_;

  // Shared memory storage containing mbarriers for software emulation
  class SharedStorage {
   public:
    // Conditional barriers are used to signal the consumer that the stage contains valid data. For
    // software emulation, this barrier will be explicitly arrived on by the producer threads. From
    // the consumers perspective, there should be no difference between this barrier and the "full"
    // barrier in a traditional TMA pipeline.
    uint64_t mConditionalBarriers[Stages];
    // Empty barriers are used to signal the producer that the consumer is finished with a stage.
    // The consumer threads will arrive on this barrier, and the producer threads will test-wait on
    // it.
    uint64_t mEmptyBarriers[Stages];
    // Primary barriers are used to indicate if data has been loaded into a stage. This barrier is
    // only used by the producer threads. It will be passed to the TMA instructions and queried in
    // the prod-test-wait function.
    uint64_t mPrimaryBarriers[Stages];
  };

  // The regular pipeline state will be used by the consumer task
  using PipelineState = cutlass::PipelineDetail::PipelineAsyncPipelineState<Stages>;

  __device__ PipelineLamport(SharedStorage& storage, int32_t consumerArrivalCount,
                             int32_t producerArrivalCount, int32_t transactionBytes, int32_t warpId,
                             int32_t barInitWarpId)
      : mConditionalBarrierPtr(&(storage.mConditionalBarriers[0])),
        mEmptyBarrierPtr(&(storage.mEmptyBarriers[0])),
        mPrimaryBarrierPtr(&(storage.mPrimaryBarriers[0])),
        mTransactionBytes(transactionBytes),
        mProducerArrivalCount(producerArrivalCount),
        mConsumerArrivalCount(consumerArrivalCount) {
    if (warpId == barInitWarpId && cute::elect_one_sync()) {
#pragma unroll
      for (int ii = 0; ii < Stages; ++ii) {
        cuda_ptx::mbarrier_init(&mEmptyBarrierPtr[ii], mConsumerArrivalCount);
        cuda_ptx::mbarrier_init(&mConditionalBarrierPtr[ii], mProducerArrivalCount);
        cuda_ptx::mbarrier_init(&mPrimaryBarrierPtr[ii], 1);
      }
    }
    // TRT-LLM Gen will have a barrier prior to the pipeline being used, so we can skip the barrier
    // here
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Producer functions
  //
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Performs an arrive and expect TX on the given stage. For normal TMA pipelines this is fused
  // with producer-acquire, but Lamport pipelines need this decoupled
  inline __device__ void producer_arrive(int32_t const index) {
    cuda_ptx::mbarrier_arrive_expect_tx(cuda_ptx::sem_release_t{}, cuda_ptx::scope_cta_t{},
                                        cuda_ptx::space_shared_t{}, &mPrimaryBarrierPtr[index],
                                        mTransactionBytes);
  }

  // Producer commit. For emulated lamport syncs, this will need to explicitly arrive on the
  // conditional barrier, but will be a no-op with hardware support
  inline __device__ void producer_commit(int32_t const index) {
    cuda_ptx::mbarrier_arrive(&mConditionalBarrierPtr[index]);
  }
  inline __device__ void producer_commit(PipelineState const& state) {
    producer_commit(state.index());
  }

  // Returns pointer to current mbarrier for use in other instructions
  [[nodiscard]] inline __device__ uint64_t* producer_get_barrier(int32_t const index) {
    return mPrimaryBarrierPtr + index;
  }
  [[nodiscard]] inline __device__ uint64_t* producer_get_barrier(PipelineState const& state) {
    return producer_get_barrier(state.index());
  }

  // Producer test-acquire. This will do a non-blocking test-wait on the empty barrier
  [[nodiscard]] inline __device__ int32_t producer_test_acquire(int32_t const index,
                                                                uint32_t const phase) {
    return cuda_ptx::mbarrier_test_wait_parity(&mEmptyBarrierPtr[index], phase);
  }
  [[nodiscard]] inline __device__ int32_t producer_test_acquire(PipelineState const& state) {
    return producer_test_acquire(state.index(), state.phase());
  }

  // Performs a producer test-wait on the given stage. Returns true if data is loaded regardless of
  // validity status.
  [[nodiscard]] inline __device__ int32_t producer_test_wait(int32_t const index,
                                                             uint32_t const phase) {
    return cuda_ptx::mbarrier_test_wait_parity(&mPrimaryBarrierPtr[index], phase);
  }

  // Performs a producer test-wait on the given stage and checks the validity of the data
  template <int kNumBytes, uint32_t kInvalidValue>
  inline __device__ void producer_test_wait_valid(int32_t const index, uint32_t const phase,
                                                  int32_t* smemPtr,
                                                  cutlass::Array<bool, 2>& result) {
    int32_t const is_loaded = producer_test_wait(index, phase);

    int32_t is_valid = false;
    if (is_loaded) {
      // TODO Should this be part of the pipeline?
      is_valid = trtllm::dev::checkDataValidityWarp<kNumBytes, kInvalidValue>(
          &reinterpret_cast<uint32_t*>(smemPtr)[int32_t{0}]);
    }

    result[0] = bool{is_loaded};
    result[1] = bool{is_valid};
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Consumer APIs
  //
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a consumer try-wait on current stage
  [[nodiscard]] inline __device__ int32_t consumer_try_wait(PipelineState const& state) {
    return cuda_ptx::mbarrier_try_wait_parity(&mConditionalBarrierPtr[state.index()],
                                              state.phase());
  }

  // Perform a consumer wait on current stage
  inline __device__ void consumer_wait(PipelineState const& state, int32_t t = int32_t{0}) {
    while (!t) {
      t = consumer_try_wait(state);
    }
  }

  // Perform a consumer release on current stage
  inline __device__ void consumer_release(PipelineState const& state) {
    if constexpr (UseUmma) {
      cutlass::arch::umma_arrive(&mEmptyBarrierPtr[state.index()]);
    } else {
      cuda_ptx::mbarrier_arrive(&mEmptyBarrierPtr[state.index()]);
    }
  }

 private:
  // Pointer to conditional barriers.
  uint64_t* mConditionalBarrierPtr{nullptr};
  // Pointer to empty barriers.
  uint64_t* mEmptyBarrierPtr{nullptr};
  // Pointer to primary barriers.
  uint64_t* mPrimaryBarrierPtr{nullptr};
  // Expected number of bytes for primary barrier.
  int32_t mTransactionBytes{0};
  // Expected number of threads for conditional barrier.
  int32_t mProducerArrivalCount{0};
  // Expected number of threads for empty barrier.
  int32_t mConsumerArrivalCount{0};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// PipelinePayloadLamport class.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages_, bool UseUmma_>
class PipelinePayloadLamport {
 public:
  // Number of stages in pipeline.
  static constexpr uint32_t Stages = Stages_;
  // Is the consumer use umma.
  static constexpr bool UseUmma = UseUmma_;

  // Shared memory storage containing mbarriers.
  class SharedStorage {
   public:
    // Full Payload64 barriers are used to signal that:
    // 1. The producer has loaded the data into the stage, through payload barrier primary phase
    // 2. The loaded data is valid, through payload barrier conditional phase
    uint64_t mFullBarriers[Stages];
    // Empty Trans64  barriers are used to signal the producer that the consumer is finished with a
    // stage. The consumer threads will arrive on this barrier, and the producer threads will
    // test-wait on it.
    uint64_t mEmptyBarriers[Stages];
  };

  // The regular pipeline state will be used by the consumer task
  using PipelineState = cutlass::PipelineDetail::PipelineAsyncPipelineState<Stages>;

  __device__ PipelinePayloadLamport(SharedStorage& storage, int32_t consumerArrivalCount,
                                    int32_t producerArrivalCount, int32_t transactionBytes,
                                    int32_t warpId, int32_t barInitWarpId)
      : mFullBarrierPtr(&(storage.mFullBarriers[0])),
        mEmptyBarrierPtr(&(storage.mEmptyBarriers[0])),
        mTransactionBytes(transactionBytes),
        mProducerArrivalCount(producerArrivalCount),
        mConsumerArrivalCount(consumerArrivalCount) {
    if (warpId == barInitWarpId && cute::elect_one_sync()) {
#pragma unroll
      for (int ii = 0; ii < Stages; ++ii) {
        detail::mbarrier_init_payload64_compat(&mFullBarrierPtr[ii], mProducerArrivalCount);
        cuda_ptx::mbarrier_init(&mEmptyBarrierPtr[ii], mConsumerArrivalCount);
      }
    }
    // TRT-LLM Gen will have a barrier prior to the pipeline being used, so we can skip the barrier
    // here
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Producer functions
  //
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Performs an arrive and expect TX on the given stage. For normal TMA pipelines this is fused
  // with producer-acquire, but Lamport pipelines need this decoupled
  inline __device__ void producer_arrive(int32_t const index) {
    cuda_ptx::mbarrier_arrive_expect_tx(cuda_ptx::sem_release_t{}, cuda_ptx::scope_cta_t{},
                                        cuda_ptx::space_shared_t{}, &mFullBarrierPtr[index],
                                        mTransactionBytes);
  }

  // Producer commit for hardware support: no-op
  inline __device__ void producer_commit(int32_t const index) {
    // no-op
  }
  inline __device__ void producer_commit(PipelineState const& state) {
    producer_commit(state.index());
  }

  // Returns pointer to current mbarrier for use in other instructions
  [[nodiscard]] inline __device__ uint64_t* producer_get_barrier(int32_t const index) {
    return mFullBarrierPtr + index;
  }
  [[nodiscard]] inline __device__ uint64_t* producer_get_barrier(PipelineState const& state) {
    return producer_get_barrier(state.index());
  }

  // Producer test-acquire. This will do a non-blocking test-wait on the empty barrier
  [[nodiscard]] inline __device__ int32_t producer_test_acquire(int32_t const index,
                                                                uint32_t const phase) {
    return cuda_ptx::mbarrier_test_wait_parity(&mEmptyBarrierPtr[index], phase);
  }
  [[nodiscard]] inline __device__ int32_t producer_test_acquire(PipelineState const& state) {
    return producer_test_acquire(state.index(), state.phase());
  }

  // Performs a producer test-wait on the given stage. Returns true if data is loaded regardless of
  // validity status.
  [[nodiscard]] inline __device__ int32_t producer_test_wait(int32_t const index,
                                                             uint32_t const phase) {
    cutlass::Array<bool, 2> result{};
    producer_test_wait_valid(index, phase, /* smemPtr */ nullptr, result);
    return result[0];
  }

  // Performs a producer test-wait on the given stage and checks the validity of the data
  template <int kNumBytes, uint32_t kInvalidValue>
  inline __device__ void producer_test_wait_valid(int32_t const index, uint32_t const phase,
                                                  int32_t* smemPtr,
                                                  cutlass::Array<bool, 2>& result) {
    // Unused when using Payload64 barriers.
    (void)smemPtr;
    auto barrier = &mFullBarrierPtr[index];

    // * wait_complete will be true if the data has been loaded.
    // * is_report_seen will be true if the data is invalid.
    bool is_report_seen;
    bool wait_complete = detail::mbarrier_test_wait_parity_primary_compat(
        cuda_ptx::sem_acquire_t{}, cuda_ptx::scope_cta_t{}, is_report_seen, barrier, phase);

    result[0] = wait_complete;
    result[1] = !is_report_seen;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Consumer APIs
  //
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a consumer try-wait on current stage
  [[nodiscard]] inline __device__ int32_t consumer_try_wait(PipelineState const& state) {
    std::uint64_t* barrier = &mFullBarrierPtr[state.index()];
    std::uint32_t phase = state.phase();

    // Waits on the conditional phase of the barrier to change meaning data is present and valid.
    bool const wait_complete = detail::mbarrier_test_wait_parity_conditional_compat(
        cuda_ptx::sem_acquire_t{}, cuda_ptx::scope_cta_t{}, barrier, phase);

    return static_cast<bool>(wait_complete);
  }

  // Perform a consumer wait on current stage
  inline __device__ void consumer_wait(PipelineState const& state, int32_t t = int32_t{0}) {
    while (!t) {
      t = consumer_try_wait(state);
    }
  }

  // Perform a consumer release on current stage
  inline __device__ void consumer_release(PipelineState const& state) {
    if constexpr (UseUmma) {
      cutlass::arch::umma_arrive(&mEmptyBarrierPtr[state.index()]);
    } else {
      cuda_ptx::mbarrier_arrive(&mEmptyBarrierPtr[state.index()]);
    }
  }

 private:
  // Pointer to full barriers.
  uint64_t* mFullBarrierPtr{nullptr};
  // Pointer to empty barriers.
  uint64_t* mEmptyBarrierPtr{nullptr};
  // Expected number of bytes for full barrier.
  int32_t mTransactionBytes{0};
  // Expected number of threads for full barrier.
  int32_t mProducerArrivalCount{0};
  // Expected number of threads for empty barrier.
  int32_t mConsumerArrivalCount{0};
};
////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace dev
}  // namespace trtllm
