#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/arch/arch.h"

#include "../kernel/flat_options.hpp"

#include "flat/common.hpp"

namespace flat::kernel {

using namespace cute;

template <typename T1, typename T2>
constexpr T1
round_down(T1 a, T2 b) {
  return (a / b) * b;
}

constexpr std::tuple<uint32_t, uint32_t>
get_register_requirements(
    uint32_t max_threads_per_block,
    uint32_t min_blocks_per_multiprocessor,
    uint32_t num_mma_warp_groups
) {
  uint32_t reg_alloc_granularity = 8;

#ifndef FLAT_DEBUG_PRINT
  uint32_t load_registers = 40 - 2 * reg_alloc_granularity;
#else
  uint32_t load_registers = 40;
#endif
  uint32_t total_registers = round_down(64 * 1024 / min_blocks_per_multiprocessor, max_threads_per_block * reg_alloc_granularity) / cutlass::NumThreadsPerWarpGroup;
  uint32_t mma_registers   = round_down((total_registers - load_registers) / num_mma_warp_groups, reg_alloc_granularity);

  // max reg is 255, 248 round to multiple of reg_alloc_granularity;
  return {cute::min(248, load_registers), cute::min(248, mma_registers)};
}

template <
    class CollectiveMainloop,
    class TileScheduler,
    class Options>
struct FlatKernelTmaWarpSpecialized {
  using ArchTag = cutlass::arch::Sm90;

  static const int     NumLoadWarpGroups = 1;
  static constexpr int NumMmaWarpGroups  = CollectiveMainloop::NumMmaWarpGroups;

  using TileShape    = typename CollectiveMainloop::TileShape;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;

  using MainloopQPipeline = typename CollectiveMainloop::MainloopQPipeline;
  using MainloopKPipeline = typename CollectiveMainloop::MainloopKPipeline;
  using MainloopVPipeline = typename CollectiveMainloop::MainloopVPipeline;
  using MainloopOPipeline = typename CollectiveMainloop::MainloopOPipeline;

  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier                  = cutlass::OrderedSequenceBarrier<
                       StagesPerMathWarpGroup, NumMmaWarpGroups>;

  struct TensorStorage {
    typename CollectiveMainloop::SharedStorage mainloop;
  };

  struct SharedStorage {
    TensorStorage tensors;

    using QPipelineStorage = typename MainloopQPipeline::SharedStorage;
    using KPipelineStorage = typename MainloopKPipeline::SharedStorage;
    using VPipelineStorage = typename MainloopVPipeline::SharedStorage;
    using OPipelineStorage = typename MainloopOPipeline::SharedStorage;

    alignas(16) QPipelineStorage q_pipeline_storage;
    alignas(16) KPipelineStorage k_pipeline_storage;
    alignas(16) VPipelineStorage v_pipeline_storage;
    alignas(16) OPipelineStorage o_pipeline_storage;

    alignas(16) cutlass::arch::ClusterBarrier load_warp_barrier;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct VarlenProblemShape {
    int64_t const* cu_seqlens;
    int64_t        total_seqlen;
    int32_t        num_seqs;
    int32_t        num_q_heads;
    int32_t        num_k_heads;
    int32_t        num_v_heads;
    int32_t        num_o_heads;
    int32_t        head_size;  // d
  };
  using ProblemShape = VarlenProblemShape;

  struct Arguments {
    ProblemShape                           problem_size;
    typename CollectiveMainloop::Arguments mainloop;
    cutlass::KernelHardwareInfo            hw_info;
  };

  struct Params {
    ProblemShape                        problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename TileScheduler::Params      scheduler;
  };

  using QPipelineParams = typename MainloopQPipeline::Params;
  using QPipelineState  = typename cutlass::PipelineState<MainloopQPipeline::Stages>;

  using KPipelineParams = typename MainloopKPipeline::Params;
  using KPipelineState  = typename cutlass::PipelineState<MainloopKPipeline::Stages>;

  using VPipelineParams = typename MainloopVPipeline::Params;
  using VPipelineState  = typename cutlass::PipelineState<MainloopVPipeline::Stages>;

  using OPipelineParams = typename MainloopOPipeline::Params;
  using OPipelineState  = typename cutlass::PipelineState<MainloopOPipeline::Stages>;

  static const int MinBlocksPerMultiprocessor = 1;
  static const int MaxThreadsPerBlock         = (NumMmaWarpGroups + NumLoadWarpGroups) * cutlass::NumThreadsPerWarpGroup;

  static constexpr uint32_t LoadRegisterRequirement = get<0>(get_register_requirements(MaxThreadsPerBlock, MinBlocksPerMultiprocessor, NumMmaWarpGroups));
  static constexpr uint32_t MmaRegisterRequirement  = get<1>(get_register_requirements(MaxThreadsPerBlock, MinBlocksPerMultiprocessor, NumMmaWarpGroups));

  static size_t
  get_workspace_size(Arguments const& args) {
    return CollectiveMainloop::get_workspace_size(args.mainloop, args.hw_info.sm_count);
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace, cudaStream_t stream) {
    return CollectiveMainloop::initialize_workspace(args.problem_size, args.mainloop, workspace, stream);
  }

  static bool
  can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_size, args.mainloop);
  }

  static dim3
  get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler);
  }

  static dim3
  get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        args.problem_size,
        CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop, workspace),
        TileScheduler::to_underlying_arguments(args.problem_size, args.hw_info, ClusterShape{}, TileShape{})
    };
  }

  CUTE_DEVICE void
  operator()(const Params& params, char* smem) {
    enum class WarpGroupRole {
      LdSt  = 0,
      Math0 = 1,
      Math1 = 2,
    };

    enum class LdStWarpRole {
      LoadQ  = 0,
      LoadK  = 1,
      LoadV  = 2,
      StoreO = 3,
    };

    TileScheduler scheduler{params.scheduler};

    // Shared memory.
    auto& storage = *reinterpret_cast<SharedStorage*>(smem);

    int  lane_idx        = cutlass::canonical_lane_idx();
    int  warp_idx        = cutlass::canonical_warp_idx_sync();
    int  warp_idx_in_wg  = warp_idx % cutlass::NumWarpsPerWarpGroup;
    int  warp_group_idx  = cutlass::canonical_warp_group_idx();
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto ldst_warp_role  = LdStWarpRole(warp_idx_in_wg);

    int      lane_predicate        = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    static constexpr int NumMathThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;

    QPipelineParams q_pipeline_params;
    q_pipeline_params.transaction_bytes = CollectiveMainloop::LoadQBytes;
    q_pipeline_params.is_leader         = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadQ);
    q_pipeline_params.num_consumers     = NumMathThreads;

    KPipelineParams k_pipeline_params;
    k_pipeline_params.transaction_bytes = CollectiveMainloop::LoadKBytes;
    k_pipeline_params.is_leader         = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadK);
    k_pipeline_params.num_consumers     = NumMathThreads;

    VPipelineParams v_pipeline_params;
    v_pipeline_params.transaction_bytes = CollectiveMainloop::LoadVBytes;
    v_pipeline_params.is_leader         = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadV);
    v_pipeline_params.num_consumers     = NumMathThreads;

    OPipelineParams o_pipeline_params;
    o_pipeline_params.producer_arv_count = NumMathThreads;
    o_pipeline_params.consumer_arv_count = cutlass::NumThreadsPerWarp;

    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::LoadQ) {
      DPRINTF0_W("ldst_warp_role: LoadQ\n");
      q_pipeline_params.role = MainloopQPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::LoadK) {
      DPRINTF0_W("ldst_warp_role: LoadK\n");
      k_pipeline_params.role = MainloopKPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::LoadV) {
      DPRINTF0_W("ldst_warp_role: LoadV\n");
      v_pipeline_params.role = MainloopVPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::StoreO) {
      DPRINTF0_W("ldst_warp_role: StoreO\n");
      o_pipeline_params.role = MainloopOPipeline::ThreadCategory::Consumer;
    }
    if (warp_group_role == WarpGroupRole::Math0 || warp_group_role == WarpGroupRole::Math1) {
      DPRINTF0_WG("warp_group_role: MathX\n");
      q_pipeline_params.role = MainloopQPipeline::ThreadCategory::Consumer;
      k_pipeline_params.role = MainloopKPipeline::ThreadCategory::Consumer;
      v_pipeline_params.role = MainloopVPipeline::ThreadCategory::Consumer;
      o_pipeline_params.role = MainloopOPipeline::ThreadCategory::Producer;
    }

    MainloopQPipeline q_pipeline(storage.q_pipeline_storage, q_pipeline_params, ClusterShape{});
    MainloopKPipeline k_pipeline(storage.k_pipeline_storage, k_pipeline_params, ClusterShape{});
    MainloopVPipeline v_pipeline(storage.v_pipeline_storage, v_pipeline_params, ClusterShape{});
    MainloopOPipeline o_pipeline(storage.o_pipeline_storage, o_pipeline_params, /*InitBarriers=*/cute::true_type{});

    QPipelineState q_smem_pipe_read;
    QPipelineState q_smem_pipe_write = cutlass::make_producer_start_state<MainloopQPipeline>();
    KPipelineState k_smem_pipe_read;
    KPipelineState k_smem_pipe_write = cutlass::make_producer_start_state<MainloopKPipeline>();
    VPipelineState v_smem_pipe_read;
    VPipelineState v_smem_pipe_write = cutlass::make_producer_start_state<MainloopVPipeline>();
    OPipelineState o_smem_pipe_read;
    OPipelineState o_smem_pipe_write = cutlass::make_producer_start_state<MainloopOPipeline>();

    // barrier sm or cluster level for initialization
    if constexpr (size(ClusterShape{}) > 1) {
      cute::cluster_arrive_relaxed();
      cute::cluster_wait();
    } else {
      __syncthreads();
    }
    DPRINTF0_WG("warpspecialized grid initialized\n");

    CollectiveMainloop collective_mainloop;

    if (warp_group_role == WarpGroupRole::LdSt) {
      DPRINTF0_WG("LsSt warp_group_idx:%d, LoadRegisterRequirement:%d\n", warp_group_idx, LoadRegisterRequirement);
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
      if (ldst_warp_role == LdStWarpRole::LoadQ) {
        auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
        CUTE_NO_UNROLL
        for (; work_desc.is_valid(params.scheduler); work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
          DPRINTF0_WG("LsSt working on LoadQ, seq_idx:%d, q_head_idx:%d, seq_len:%lld)\n",
                      work_desc.seq_idx, work_desc.q_head_idx(), work_desc.seq_len);
          auto tile_shape = typename CollectiveMainloop::TileShape{};
          collective_mainloop.template load<CollectiveMainloop::LoadQ>(
              params.mainloop.tma_load_q, params.problem_size, tile_shape, work_desc,
              q_pipeline, q_smem_pipe_write, storage.tensors.mainloop.smem_q
          );
        }
      } else if (ldst_warp_role == LdStWarpRole::LoadK) {
        auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
        CUTE_NO_UNROLL
        for (; work_desc.is_valid(params.scheduler); work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
          DPRINTF0_WG("LsSt working on LoadK, seq_idx:%d, k_head_idx:%d, seq_len:%lld)\n",
                      work_desc.seq_idx, work_desc.k_head_idx(), work_desc.seq_len);
          auto tile_shape = typename CollectiveMainloop::TileShape{};
          collective_mainloop.template load<CollectiveMainloop::LoadK>(
              params.mainloop.tma_load_k, params.problem_size, tile_shape, work_desc,
              k_pipeline, k_smem_pipe_write, storage.tensors.mainloop.smem_k
          );
        }
      } else if (ldst_warp_role == LdStWarpRole::LoadV) {
        auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
        CUTE_NO_UNROLL
        for (; work_desc.is_valid(params.scheduler); work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
          DPRINTF0_WG("LsSt working on LoadV, seq_idx:%d, v_head_idx:%d, seq_len:%lld)\n",
                      work_desc.seq_idx, work_desc.v_head_idx(), work_desc.seq_len);
          auto tile_shape = typename CollectiveMainloop::TileShape{};
          collective_mainloop.template load<CollectiveMainloop::LoadV>(
              params.mainloop.tma_load_v, params.problem_size, tile_shape, work_desc,
              v_pipeline, v_smem_pipe_write, storage.tensors.mainloop.smem_v
          );
        }
      } else if (ldst_warp_role == LdStWarpRole::StoreO) {
        auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
        DPRINTF0_WG("LsSt working on StoreO, seq_idx:%d, o_head_idx:%d seq_len:%lld)\n",
                    work_desc.seq_idx, work_desc.o_head_idx(), work_desc.seq_len);
        auto tile_shape = typename CollectiveMainloop::TileShape{};
        collective_mainloop.store(
            params.mainloop.tma_store_o, params.mainloop.tensormaps, params.problem_size, tile_shape, work_desc,
            o_pipeline, o_smem_pipe_read, storage.tensors.mainloop.smem_o
        );
      }
    } else if (warp_group_role == WarpGroupRole::Math0 || warp_group_role == WarpGroupRole::Math1) {
      DPRINTF0_WG("Consumer warp_group_idx:%d, MmaRegisterRequirement:%d\n", warp_group_idx, MmaRegisterRequirement);
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
      auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
      CUTE_NO_UNROLL
      for (; work_desc.is_valid(params.scheduler); work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
        DPRINTF0_WG("Consumer, seq_idx:%d, q/k/v/o_head_idx:(%d,%d,%d,%d), seq_len:%lld)\n",
                    work_desc.seq_idx, work_desc.q_head_idx(), work_desc.k_head_idx(), work_desc.v_head_idx(), work_desc.o_head_idx(), work_desc.seq_len);
        collective_mainloop.compute(
            params.mainloop, params.problem_size, work_desc,
            q_pipeline, q_smem_pipe_read,
            k_pipeline, k_smem_pipe_read,
            v_pipeline, v_smem_pipe_read,
            o_pipeline, o_smem_pipe_write,
            storage.tensors.mainloop
        );
      }
    } else {
      DPRINTF0_WG("Unknown warp role, warp_group_idx:%d\n", warp_group_idx);
    }

    __syncthreads();
  }
};

}  // namespace flat::kernel
