/*
 * Copyright (c) 2026 by FlashInfer team.
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

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/cutlass.h"
#include "cutlass/pipeline/pipeline.hpp"

// Reuse hopper code
#include "flashinfer/flat/common.hpp"
#include "flashinfer/flat/hopper/kernel/flat_options.hpp"
#include "flashinfer/flat/unused.hpp"

namespace flat::kernel {

using namespace cute;

template <typename T1, typename T2>
constexpr T1 round_down(T1 a, T2 b) {
  return (a / b) * b;
}

constexpr std::tuple<uint32_t, uint32_t> get_register_requirements(
    uint32_t max_threads_per_block, uint32_t min_blocks_per_multiprocessor,
    uint32_t num_mma_warp_groups) {
  uint32_t reg_alloc_granularity = 8;

#if !defined(FLAT_DEBUG_PRINT) || !FLAT_DEBUG_PRINT
  uint32_t load_registers = 40 - 2 * reg_alloc_granularity;
#else
  uint32_t load_registers = 40;
#endif
  uint32_t total_registers = round_down(64 * 1024 / min_blocks_per_multiprocessor,
                                        max_threads_per_block * reg_alloc_granularity) /
                             cutlass::NumThreadsPerWarpGroup;
  uint32_t mma_registers =
      round_down((total_registers - load_registers) / num_mma_warp_groups, reg_alloc_granularity);

  // max reg is 255, 248 round to multiple of reg_alloc_granularity;
  return {cute::min(248, load_registers), cute::min(248, mma_registers)};
}

template <class CollectiveMainloop, class TileScheduler, class Options>
struct FlatKernelTmaWarpSpecializedDeltaRule {
  using ArchTag = cutlass::arch::Sm120;

  static const int NumLoadWarpGroups = 1;
  static constexpr int NumMmaWarpGroups = CollectiveMainloop::NumMmaWarpGroups;

  static constexpr int NeedsAlpha = CollectiveMainloop::NeedsAlpha;
  static constexpr int NeedsBeta = CollectiveMainloop::NeedsBeta;

  using TileShape = typename CollectiveMainloop::TileShape;
  using ClusterShape = typename CollectiveMainloop::ClusterShape;

  using MainloopQPipeline = typename CollectiveMainloop::MainloopQPipeline;
  using MainloopKPipeline = typename CollectiveMainloop::MainloopKPipeline;
  using MainloopVPipeline = typename CollectiveMainloop::MainloopVPipeline;
  using MainloopOPipeline = typename CollectiveMainloop::MainloopOPipeline;

  using MainloopAlphaPipeline = typename CollectiveMainloop::MainloopAlphaPipeline;
  using MainloopBetaPipeline = typename CollectiveMainloop::MainloopBetaPipeline;

  using OrderedMathBarriers = typename CollectiveMainloop::OrderedMathBarriers;

  static constexpr uint32_t StagesPerMathWarpGroup = 2;

  // FIXME: remove this after moving to HMMA
  using MathWarpGroupOrderBarrier =
      cutlass::OrderedSequenceBarrier<StagesPerMathWarpGroup, NumMmaWarpGroups>;

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

    using AlphaPipelineStorage = typename MainloopAlphaPipeline::SharedStorage;
    using BetaPipelineStorage = typename MainloopBetaPipeline::SharedStorage;
    alignas(16) AlphaPipelineStorage alpha_pipeline_storage;
    alignas(16) BetaPipelineStorage beta_pipeline_storage;

    alignas(16) cutlass::arch::ClusterBarrier load_warp_barrier;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct VarlenProblemShape {
    int64_t const* cu_seqlens;
    int64_t total_seqlen;
    int32_t num_seqs;
    int32_t num_q_heads;
    int32_t num_k_heads;
    int32_t num_v_heads;
    int32_t num_o_heads;
    int32_t num_sab_heads;  // state, alpha, beta
    int32_t head_size;      // d
  };
  using ProblemShape = VarlenProblemShape;

  struct Arguments {
    ProblemShape problem_size;
    typename CollectiveMainloop::Arguments mainloop;
    cutlass::KernelHardwareInfo hw_info;
  };

  struct Params {
    ProblemShape problem_size;
    typename CollectiveMainloop::Params mainloop;
    typename TileScheduler::Params scheduler;
  };

  using QPipelineParams = typename MainloopQPipeline::Params;
  using QPipelineState = typename cutlass::PipelineState<MainloopQPipeline::Stages>;

  using KPipelineParams = typename MainloopKPipeline::Params;
  using KPipelineState = typename cutlass::PipelineState<MainloopKPipeline::Stages>;

  using VPipelineParams = typename MainloopVPipeline::Params;
  using VPipelineState = typename cutlass::PipelineState<MainloopVPipeline::Stages>;

  using OPipelineParams = typename MainloopOPipeline::Params;
  using OPipelineState = typename cutlass::PipelineState<MainloopOPipeline::Stages>;

  using AlphaPipelineParams =
      std::conditional_t<NeedsAlpha, typename MainloopAlphaPipeline::Params, Unused>;
  using AlphaPipelineState =
      std::conditional_t<NeedsAlpha, cutlass::PipelineState<MainloopAlphaPipeline::Stages>, Unused>;

  using BetaPipelineParams =
      std::conditional_t<NeedsBeta, typename MainloopBetaPipeline::Params, Unused>;
  using BetaPipelineState =
      std::conditional_t<NeedsBeta, cutlass::PipelineState<MainloopBetaPipeline::Stages>, Unused>;

  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr int MaxThreadsPerBlock =
      (NumLoadWarpGroups + NumMmaWarpGroups) * cutlass::NumThreadsPerWarpGroup;

  static constexpr auto RegisterRequirements =
      get_register_requirements(MaxThreadsPerBlock, MinBlocksPerMultiprocessor, NumMmaWarpGroups);
  static constexpr uint32_t LdStRegisterRequirement = get<0>(RegisterRequirements);
  static constexpr uint32_t MmaRegisterRequirement = get<1>(RegisterRequirements);

  static size_t get_workspace_size(Arguments const& args) {
    return CollectiveMainloop::get_workspace_size(args.mainloop, args.hw_info.sm_count);
  }

  static cutlass::Status initialize_workspace(Arguments const& args, void* workspace,
                                              cudaStream_t stream) {
    return CollectiveMainloop::initialize_workspace(args.problem_size, args.mainloop, workspace,
                                                    stream);
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.problem_size, args.mainloop);
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::get_grid_shape(params.scheduler);
  }

  static dim3 get_block_shape() {
    dim3 block(MaxThreadsPerBlock, 1, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{
        args.problem_size,
        CollectiveMainloop::to_underlying_arguments(args.problem_size, args.mainloop, workspace),
        TileScheduler::to_underlying_arguments(args.problem_size, args.hw_info, ClusterShape{},
                                               TileShape{})};
  }

  CUTE_DEVICE void operator()(const Params& params, char* smem) {
    enum class WarpGroupRole {
      LdSt = 0,
      Math0 = 1,
      Math1 = 2,
    };

    // NOTE: CollectiveInverse will have more utilization on warp 0&1
    //       so we put beta and alpha preprocessing on warp 2&3
    enum class LdStWarpRole {
      LoadQKV = 0,
      StoreO = 1,
      LoadBeta = 2,
      LoadAlpha = 3,
    };

    TileScheduler scheduler{params.scheduler};

    // Shared memory.
    auto& storage = *reinterpret_cast<SharedStorage*>(smem);

    int lane_idx = cutlass::canonical_lane_idx();
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int warp_idx_in_wg = warp_idx % cutlass::NumWarpsPerWarpGroup;
    int warp_group_idx = cutlass::canonical_warp_group_idx();
    auto warp_group_role = WarpGroupRole(warp_group_idx);
    auto ldst_warp_role = LdStWarpRole(warp_idx_in_wg);

    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }

    constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;

    QPipelineParams q_pipeline_params;
    q_pipeline_params.transaction_bytes = CollectiveMainloop::LoadQBytes;
    q_pipeline_params.is_leader = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadQKV);
    q_pipeline_params.num_consumers = NumMmaThreads;

    KPipelineParams k_pipeline_params;
    k_pipeline_params.transaction_bytes = CollectiveMainloop::LoadKBytes;
    k_pipeline_params.is_leader = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadQKV);
    k_pipeline_params.num_consumers = NumMmaThreads;

    VPipelineParams v_pipeline_params;
    v_pipeline_params.transaction_bytes = CollectiveMainloop::LoadVBytes;
    v_pipeline_params.is_leader = lane_predicate && (ldst_warp_role == LdStWarpRole::LoadQKV);
    v_pipeline_params.num_consumers = NumMmaThreads;

    OPipelineParams o_pipeline_params;
    o_pipeline_params.producer_arv_count = NumMmaThreads;
    o_pipeline_params.consumer_arv_count = cutlass::NumThreadsPerWarp;

    AlphaPipelineParams alpha_pipeline_params;
    if constexpr (NeedsAlpha) {
      alpha_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp;
      alpha_pipeline_params.consumer_arv_count = NumMmaThreads;
    }

    BetaPipelineParams beta_pipeline_params;
    if constexpr (NeedsBeta) {
      beta_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp;
      beta_pipeline_params.consumer_arv_count = NumMmaThreads;
    }

    OrderedMathBarriers math_barriers;

    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::LoadQKV) {
      DPRINTF0_W("ldst_warp_role: LoadQKV\n");
      q_pipeline_params.role = MainloopQPipeline::ThreadCategory::Producer;
      k_pipeline_params.role = MainloopKPipeline::ThreadCategory::Producer;
      v_pipeline_params.role = MainloopVPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::StoreO) {
      DPRINTF0_W("ldst_warp_role: StoreO\n");
      o_pipeline_params.role = MainloopOPipeline::ThreadCategory::Consumer;
    }
    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::LoadBeta) {
      if constexpr (NeedsBeta) {
        beta_pipeline_params.role = MainloopBetaPipeline::ThreadCategory::Producer;
      }
    }
    if (warp_group_role == WarpGroupRole::LdSt && ldst_warp_role == LdStWarpRole::LoadAlpha) {
      if constexpr (NeedsAlpha) {
        alpha_pipeline_params.role = MainloopAlphaPipeline::ThreadCategory::Producer;
      }
    }
    if (warp_group_role == WarpGroupRole::Math0 || warp_group_role == WarpGroupRole::Math1) {
      DPRINTF0_WG("warp_group_role: MathX\n");
      q_pipeline_params.role = MainloopQPipeline::ThreadCategory::Consumer;
      k_pipeline_params.role = MainloopKPipeline::ThreadCategory::Consumer;
      v_pipeline_params.role = MainloopVPipeline::ThreadCategory::Consumer;
      o_pipeline_params.role = MainloopOPipeline::ThreadCategory::Producer;

      if constexpr (NeedsAlpha) {
        alpha_pipeline_params.role = MainloopAlphaPipeline::ThreadCategory::Consumer;
      }

      math_barriers.init(warp_group_idx - 1);
    }

    MainloopQPipeline q_pipeline(storage.q_pipeline_storage, q_pipeline_params, ClusterShape{});
    MainloopKPipeline k_pipeline(storage.k_pipeline_storage, k_pipeline_params, ClusterShape{});
    MainloopVPipeline v_pipeline(storage.v_pipeline_storage, v_pipeline_params, ClusterShape{});
    MainloopOPipeline o_pipeline(storage.o_pipeline_storage, o_pipeline_params,
                                 /*InitBarriers=*/cute::true_type{});

    MainloopAlphaPipeline alpha_pipeline(storage.alpha_pipeline_storage, alpha_pipeline_params,
                                         /*InitBarriers=*/cute::true_type{});
    MainloopBetaPipeline beta_pipeline(storage.beta_pipeline_storage, beta_pipeline_params,
                                       /*InitBarriers=*/cute::true_type{});

    QPipelineState q_smem_pipe_read;
    QPipelineState q_smem_pipe_write = cutlass::make_producer_start_state<MainloopQPipeline>();
    KPipelineState k_smem_pipe_read;
    KPipelineState k_smem_pipe_write = cutlass::make_producer_start_state<MainloopKPipeline>();
    VPipelineState v_smem_pipe_read;
    VPipelineState v_smem_pipe_write = cutlass::make_producer_start_state<MainloopVPipeline>();
    OPipelineState o_smem_pipe_read;
    OPipelineState o_smem_pipe_write = cutlass::make_producer_start_state<MainloopOPipeline>();

    AlphaPipelineState alpha_smem_pipe_read;
    AlphaPipelineState alpha_smem_pipe_write;
    if constexpr (NeedsAlpha) {
      alpha_smem_pipe_write = cutlass::make_producer_start_state<MainloopAlphaPipeline>();
    }
    BetaPipelineState beta_smem_pipe_read;
    BetaPipelineState beta_smem_pipe_write;
    if constexpr (NeedsBeta) {
      beta_smem_pipe_write = cutlass::make_producer_start_state<MainloopBetaPipeline>();
    }

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
      DPRINTF0_WG("LsSt warp_group_idx:%d, RegisterRequirement:%d\n", warp_group_idx,
                  LdStRegisterRequirement);
      cutlass::arch::warpgroup_reg_dealloc<LdStRegisterRequirement>();
      if (ldst_warp_role == LdStWarpRole::LoadQKV) {
        auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
        CUTE_NO_UNROLL
        for (; work_desc.is_valid(params.scheduler);
             work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
          DPRINTF0_WG(
              "LsSt working on LoadQ/K/V, seq_idx:%d, q/k/v_head_idx:(%d,%d,%d), seq_len:%lld)\n",
              work_desc.seq_idx, work_desc.q_head_idx(), work_desc.k_head_idx(),
              work_desc.v_head_idx(), work_desc.seq_len);
          auto tile_shape = typename CollectiveMainloop::TileShape{};
          collective_mainloop.load_qkv(params.mainloop, params.problem_size, tile_shape, work_desc,
                                       q_pipeline, q_smem_pipe_write, k_pipeline, k_smem_pipe_write,
                                       v_pipeline, v_smem_pipe_write, storage.tensors.mainloop);
        }
      } else if (ldst_warp_role == LdStWarpRole::LoadBeta) {
        if constexpr (NeedsBeta) {
          auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
          CUTE_NO_UNROLL
          for (; work_desc.is_valid(params.scheduler);
               work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
            DPRINTF0_WG("LsSt working on LoadBeta, seq_idx:%d, sab_head_idx:%d, seq_len:%lld)\n",
                        work_desc.seq_idx, work_desc.o_head_idx(), work_desc.seq_len);
            auto tile_shape = typename CollectiveMainloop::TileShape{};
            collective_mainloop.load_beta(params.mainloop, params.problem_size, tile_shape,
                                          work_desc, beta_pipeline, beta_smem_pipe_write,
                                          storage.tensors.mainloop);
          }
        }
      } else if (ldst_warp_role == LdStWarpRole::LoadAlpha) {
        if constexpr (NeedsAlpha) {
          auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
          CUTE_NO_UNROLL
          for (; work_desc.is_valid(params.scheduler);
               work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
            DPRINTF0_WG("LsSt working on LoadAlpha, seq_idx:%d, sab_head_idx:%d, seq_len:%lld)\n",
                        work_desc.seq_idx, work_desc.o_head_idx(), work_desc.seq_len);
            auto tile_shape = typename CollectiveMainloop::TileShape{};
            collective_mainloop.load_alpha(params.mainloop, params.problem_size, tile_shape,
                                           work_desc, alpha_pipeline, alpha_smem_pipe_write,
                                           storage.tensors.mainloop);
          }
        }
      } else if (ldst_warp_role == LdStWarpRole::StoreO) {
        auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
        DPRINTF0_WG("LsSt working on StoreO, seq_idx:%d, o_head_idx:%d, seq_len:%lld)\n",
                    work_desc.seq_idx, work_desc.o_head_idx(), work_desc.seq_len);
        auto tile_shape = typename CollectiveMainloop::TileShape{};
        collective_mainloop.store(params.mainloop.tma_store_o, params.mainloop.tensormaps,
                                  params.problem_size, tile_shape, work_desc, o_pipeline,
                                  o_smem_pipe_read, storage.tensors.mainloop.smem_o);
      }
    } else if (warp_group_role == WarpGroupRole::Math0 || warp_group_role == WarpGroupRole::Math1) {
      DPRINTF0_WG("Compute[state]: warp_group_idx:%d, RegisterRequirement:%d\n", warp_group_idx,
                  StateMmaRegisterRequirement);
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
      auto work_desc = scheduler.get_next_work(params.scheduler, params.problem_size);
      CUTE_NO_UNROLL
      for (; work_desc.is_valid(params.scheduler);
           work_desc = scheduler.get_next_work(params.scheduler, params.problem_size)) {
        DPRINTF0_WG("Compute[state]: seq_idx:%d, qk/v/o_head_idx:(%d,%d,%d,%d), seq_len:%lld)\n",
                    work_desc.seq_idx, work_desc.q_head_idx(), work_desc.k_head_idx(),
                    work_desc.v_head_idx(), work_desc.o_head_idx(), work_desc.seq_len);
        collective_mainloop.compute(params.mainloop, params.problem_size, work_desc, q_pipeline,
                                    q_smem_pipe_read, k_pipeline, k_smem_pipe_read, v_pipeline,
                                    v_smem_pipe_read, o_pipeline, o_smem_pipe_write, alpha_pipeline,
                                    alpha_smem_pipe_read, beta_pipeline, beta_smem_pipe_read,
                                    math_barriers, storage.tensors.mainloop);
      }
    } else {
      DPRINTF0_WG("Unknown warp role, warp_group_idx:%d\n", warp_group_idx);
    }

    __syncthreads();
  }
};

}  // namespace flat::kernel
