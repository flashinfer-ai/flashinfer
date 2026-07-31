/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// FusedAttnFwdSm100 — kernel class with SharedStorage, operator(), warp dispatch
// Step 2: 4 WG (512 threads) — 2 Softmax WGs + Correction WG + MMA/Load/Epi WG
#pragma once

// Enable setmaxnreg on SM100a (CUDA 13.1 doesn't define __CUDA_ARCH_FEAT_SM100_ALL)
// Must be defined before cutlass/arch/reg_reconfig.h is included.
#ifndef CUDA_CTA_RECONFIG_ACTIVATED
#define CUDA_CTA_RECONFIG_ACTIVATED 1
#endif

#include <math_constants.h>

#include "cute/arch/tmem_allocator_sm100.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/detail/sm100_tmem_helper.hpp"
#include "epilogue_fwd_sm100.hpp"
#include "mainloop_fwd_sm100.hpp"
#include "pipeline.hpp"
#include "tile_scheduler.hpp"
#include "utils.h"

namespace flash {

namespace cute = ::cute;

#define CUTLASS_ARCH_MMA_SM100_SUPPORTED 1

// Set warp register count to target N (from cutlass example 77)
template <uint32_t N>
__device__ __forceinline__ void warpgroup_reg_set() {
  if constexpr (N < 128) {
    cutlass::arch::warpgroup_reg_dealloc<N>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<N>();
  }
}

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <typename CollectiveMainloop_, typename CollectiveEpilogue_>
struct FusedAttnFwdSm100 {
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;

  // Re-export commonly used types/constants from mainloop
  using ElementA = typename CollectiveMainloop::ElementA;
  using ElementB = typename CollectiveMainloop::ElementB;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;

  static constexpr int kRows = CollectiveMainloop::kRows;
  static constexpr int kQkK = CollectiveMainloop::kQkK;
  static constexpr int kOutputCols = CollectiveMainloop::kOutputCols;
  static constexpr int kDualCols = CollectiveMainloop::kDualCols;
  static constexpr int kQkN = CollectiveMainloop::kQkN;

  // 512 threads = 16 warps = 4 warpgroups
  static constexpr int kThreads = 512;

  // WG3 (warps 12-15): MMA / Load / Epilogue / Scheduler
  static constexpr int kMmaWarp = 12;
  static constexpr int kEpiWarp = 13;
  static constexpr int kLoadWarp = 14;
  static constexpr int kSchedWarp = 15;

  // Worker threads: all except the scheduler warp (kSchedWarp)
  static constexpr int kWorkerThreads = kThreads - 32;  // 480

  static constexpr int kSoftmaxWarps = CollectiveMainloop::kSoftmaxWarps;
  static constexpr int kCorrWarps = CollectiveMainloop::kCorrWarps;

  // Register allocation: correction needs extra regs for combine math
  // 184*128*2 + 88*128 + 56*128 = 47104 + 11264 + 7168 = 65536 regs/SM
  static constexpr int kRegsSoftmax = 184;
  static constexpr int kRegsCorrection = 88;
  static constexpr int kRegsOther = 56;

  // ---- Named barriers (used by both mainloop softmax and mainloop correction) ----
  // IDs 0-7: per-warp sm_stats (BSA pattern: bar.arrive + bar.sync for SMEM visibility).
  //   stage(0,1) x warp(0,1,2,3) = 8 barriers, 64 threads each (32 softmax + 32 correction).
  //   Softmax: bar.arrive (non-blocking), Correction: bar.arrive_and_wait (blocking).
  // IDs 4-5 reused for correction reduce (they don't overlap in time with sm_stats IDs 4-5).
  enum NamedBarriers : int {
    SmStatsNotify = 0,  // +(stage*4+warp_idx): IDs 0..7
                        // 64 threads each: 32 softmax + 32 correction
                        // Softmax: bar.arrive (non-blocking), Correction: bar.sync (blocking)
    Reduce_02 = 4,      // Reused after sm_stats done. Correction warp-pair (8,10)
    Reduce_13 = 5,      // Reused after sm_stats done. Correction warp-pair (9,11)
  };

  // ---- SharedStorage ----
  struct SharedStorage {
    struct TensorStorage {
      union {
        typename CollectiveMainloop::TensorStorage mainloop;
        typename CollectiveEpilogue::TensorStorage epilogue;
      };
    } tensors;

    struct PipelineStorage {
      PipelineKV::SharedStorage kv;
      PipelineSPO::SharedStorage spo;
      PipelineOAcc::SharedStorage o_acc;
      PipelineOEpi::SharedStorage o_epi;
      PipelineSmStats::SharedStorage sm_stats;
      PipelinePLastSplit::SharedStorage p_lastsplit;
      PipelineCLC::SharedStorage clc;
      alignas(16) CLCResponse clc_response[kCLCStages];

      alignas(16) cute::uint64_t bar_q_ready;
    } pipelines;

    alignas(16) cute::uint32_t tmem_base_ptr;
    alignas(4) int tmem_ready;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static_assert(SharedStorageSize <= 228 * 1024, "SharedStorage exceeds SM100 228KB SMEM limit");

  // ---- Arguments (host-side) ----
  struct Arguments {
    typename CollectiveMainloop::Arguments mainloop;
    typename CollectiveEpilogue::Arguments epilogue;
    int rows_padded;
    int seq_padded;  // virtual: num_kv_iters * kDualCols (determines pipeline iterations)
    int heads;
    int batch = 1;
    int seq_kv = 0;  // actual K/V seq length for TMA descriptor (0 = use seq_padded)
  };

  // ---- Params (device-side) ----
  struct Params {
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
    int num_row_tiles;
    int num_kv_blocks;
    int num_heads;
    int batch;
  };

  // ---- Convert Arguments -> Params ----
  static Params to_underlying_arguments(Arguments const& args) {
    int const num_row_tiles = args.rows_padded / kRows;
    int const num_kv_blocks = args.seq_padded / kDualCols;

    int seq_kv = args.seq_kv > 0 ? args.seq_kv : args.seq_padded;
    auto mainloop_params = CollectiveMainloop::to_underlying_arguments(
        args.mainloop, args.rows_padded, args.seq_padded, args.heads, seq_kv, args.batch);
    auto epilogue_params = CollectiveEpilogue::to_underlying_arguments(
        args.epilogue, args.rows_padded, args.heads, args.batch);

    return {mainloop_params, epilogue_params, num_row_tiles, num_kv_blocks, args.heads, args.batch};
  }

  static dim3 get_grid_shape(Params const& params) {
    return dim3(params.num_row_tiles, params.num_heads, params.batch);
  }

  static dim3 get_block_shape() { return dim3(kThreads, 1, 1); }

  // ---- operator(): CLC persistent scheduling, TMEM alloc/free, pipeline init, warp dispatch ----
  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using cutlass::arch::NamedBarrier;

    auto& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const int global_num_kv_blocks = params.num_kv_blocks;

    // ======== Phase 1: Pipeline & barrier init ========
    // PipelineKV (CUTLASS PipelineTmaUmmaAsync): example 77 pattern.
    // Params setup before construction.
    int lane_predicate = cute::elect_one_sync();
    PipelineKV::Params pipeline_kv_params;
    pipeline_kv_params.transaction_bytes = CollectiveMainloop::kKVBytes;
    pipeline_kv_params.role = (warp_idx == kLoadWarp)  ? PipelineKV::ThreadCategory::Producer
                              : (warp_idx == kMmaWarp) ? PipelineKV::ThreadCategory::Consumer
                                                       : PipelineKV::ThreadCategory::NonParticipant;
    pipeline_kv_params.is_leader = lane_predicate && (warp_idx == kLoadWarp);
    pipeline_kv_params.num_consumers = 1;  // MMA warp single-thread consumer_release

    // Construct with barriers and masks deferred
    PipelineKV pipeline_kv(shared_storage.pipelines.kv, pipeline_kv_params,
                           cute::Shape<cute::_1, cute::_1, cute::_1>{}, cute::false_type{},
                           cute::false_type{});

    PipelineSPO pipeline_s_p_o(shared_storage.pipelines.spo);
    PipelineOAcc pipeline_o_acc(shared_storage.pipelines.o_acc);
    PipelineOEpi pipeline_o_epi(shared_storage.pipelines.o_epi);
    PipelineSmStats pipeline_sm_stats(shared_storage.pipelines.sm_stats);
    PipelinePLastSplit pipeline_p_lastsplit(shared_storage.pipelines.p_lastsplit);

    // PipelineCLC: CUTLASS PipelineCLCFetchAsync constructor initializes barriers.
    // Scheduler warp = ProducerConsumer, all others = Consumer.
    PipelineCLC::Params clc_params;
    clc_params.transaction_bytes = 16;  // sizeof(CLCResponse)
    clc_params.role = (warp_idx == kSchedWarp) ? PipelineCLC::ThreadCategory::ProducerConsumer
                                               : PipelineCLC::ThreadCategory::Consumer;
    clc_params.is_leader = 1;           // single CTA
    clc_params.num_consumers = 1;       // single CTA
    clc_params.producer_blockid = 0;    // single CTA
    clc_params.producer_arv_count = 1;  // lane 0 does arrive_and_expect_tx
    clc_params.consumer_arv_count =
        kWorkerThreads;  // 480 worker threads do consumer_release (scheduler does NOT)
    clc_params.initializing_warp = 0;  // warp 0 initializes CLC barriers
    PipelineCLC pipeline_clc(shared_storage.pipelines.clc, clc_params);

    // Init all barriers: KV pipeline (warp-wide) + others (elect_one)
    if (warp_idx == 0) {
      PipelineKV::init_barriers(shared_storage.pipelines.kv, pipeline_kv_params,
                                cute::Shape<cute::_1, cute::_1, cute::_1>{});
    }
    if (warp_idx == 0 && lane_predicate) {
      pipeline_s_p_o.init();
      pipeline_o_acc.init();
      pipeline_o_epi.init();
      pipeline_sm_stats.init();
      pipeline_p_lastsplit.init();
      // PipelineCLC barriers initialized by CUTLASS constructor above
      cute::initialize_barrier(shared_storage.pipelines.bar_q_ready, 1);
      shared_storage.tmem_ready = 0;
    }
    fence_barrier_init();
    __syncthreads();
    pipeline_kv.init_masks(cute::Shape<cute::_1, cute::_1, cute::_1>{});
    pipeline_s_p_o.precompute_addrs();
    pipeline_o_acc.precompute_addrs();
    pipeline_o_epi.precompute_addrs();
    pipeline_sm_stats.precompute_addrs();
    pipeline_p_lastsplit.precompute_addrs();

    // ======== Phase 2: Prefetch TMA descriptors (example 77 / FA hopper pattern) ========
    if (warp_idx == kLoadWarp && elect_one_sync()) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    }
    if (warp_idx == kEpiWarp && elect_one_sync()) {
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // ======== Phase 3: WG3 reg set (warps 12-15 together, before dispatch) ========
    if (warp_idx >= 12) {
      warpgroup_reg_set<kRegsOther>();
    }

    // ======== Phase 4: Warp dispatch ========
    using TmemAllocator = TMEM::Allocator1Sm;
    TmemAllocator tmem_alloc{};
    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    if (warp_idx == kSchedWarp) {
      // ===== WG3 warp 15: CLC scheduler (lane 0 only) =====
      // Only lane 0 runs the scheduling loop (matching original pattern).
      // producer_tail drains the pipeline on exit.
      if (lane_idx == 0) {
        CLCTileScheduler tile_sched(
            pipeline_clc, shared_storage.pipelines.clc_response,
            {params.num_row_tiles, params.num_kv_blocks, params.num_heads, params.batch});

        auto work = tile_sched.initial_work_tile_info();
        while (work.is_valid) {
          tile_sched.advance_to_next_work([&]() {
            pipeline_o_epi.init();
            fence_barrier_init();
          });
          work = tile_sched.fetch_next_work();
        }
        // producer_tail intentionally omitted (matches original)
      }
    } else if (warp_idx == kMmaWarp) {
      // ===== WG3 warp 12: MMA (TMEM alloc here) =====
      tmem_alloc.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);
      tmem_alloc.release_allocation_lock();
      __threadfence_block();
      *reinterpret_cast<volatile int*>(&shared_storage.tmem_ready) = 1;

      const uint32_t tmem_base = shared_storage.tmem_base_ptr;
      typename CollectiveMainloop::MmaState mma_state;
      CLCTileScheduler tile_sched(
          pipeline_clc, shared_storage.pipelines.clc_response,
          {params.num_row_tiles, params.num_kv_blocks, params.num_heads, params.batch});

      auto work = tile_sched.initial_work_tile_info();
      while (work.is_valid) {
        int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
            params.mainloop, work.batch, work.head, work.row_tile, global_num_kv_blocks);
        mma_state = mainloop.mma(pipeline_kv, pipeline_s_p_o, pipeline_o_acc, pipeline_p_lastsplit,
                                 shared_storage, tmem_base, tile_nkv, mma_state);
        work = tile_sched.consumer_advance();
      }

      tmem_alloc.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
    } else if (warp_idx == kEpiWarp) {
      // ===== WG3 warp 13: Epilogue TMA store =====
      while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {
      }
      __threadfence_block();

      typename CollectiveEpilogue::EpiState epi_state;
      CLCTileScheduler tile_sched(
          pipeline_clc, shared_storage.pipelines.clc_response,
          {params.num_row_tiles, params.num_kv_blocks, params.num_heads, params.batch});

      auto work = tile_sched.initial_work_tile_info();
      while (work.is_valid) {
        if (elect_one_sync()) {
          pipeline_o_epi.prefill();
        }
        epilogue.template tma_store<SharedStorage>(params.epilogue, pipeline_o_epi, shared_storage,
                                                   work.head, work.row_tile, work.batch,
                                                   params.num_row_tiles, epi_state);
        epi_state = typename CollectiveEpilogue::EpiState{};
        work = tile_sched.consumer_advance();
      }
    } else if (warp_idx == kLoadWarp) {
      // ===== WG3 warp 14: Load TMA =====
      while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {
      }
      __threadfence_block();

      // Producer start state: phase=1 (no prefill needed, example 77 pattern)
      typename CollectiveMainloop::LoadState load_state;
      CLCTileScheduler tile_sched(
          pipeline_clc, shared_storage.pipelines.clc_response,
          {params.num_row_tiles, params.num_kv_blocks, params.num_heads, params.batch});

      auto work = tile_sched.initial_work_tile_info();
      while (work.is_valid) {
        int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
            params.mainloop, work.batch, work.head, work.row_tile, global_num_kv_blocks);
        int raw_bc = CollectiveMainloop::get_tile_raw_block_count(params.mainloop, work.batch,
                                                                  work.head, work.row_tile);
        load_state =
            mainloop.load(params.mainloop, pipeline_kv, shared_storage, work.head, work.row_tile,
                          work.batch, params.num_row_tiles, tile_nkv, raw_bc, load_state);
        work = tile_sched.consumer_advance();
      }
    } else if (warp_idx >= 8) {
      // ===== WG2 (warps 8-11): Correction =====
      cutlass::arch::warpgroup_reg_dealloc<kRegsCorrection>();
      while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {
      }
      __threadfence_block();
      pipeline_sm_stats.prefill_consumer();

      const uint32_t tmem_base = shared_storage.tmem_base_ptr;
      typename CollectiveMainloop::CorrState corr_state;
      CLCTileScheduler tile_sched(
          pipeline_clc, shared_storage.pipelines.clc_response,
          {params.num_row_tiles, params.num_kv_blocks, params.num_heads, params.batch});

      auto work = tile_sched.initial_work_tile_info();
      while (work.is_valid) {
        int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
            params.mainloop, work.batch, work.head, work.row_tile, global_num_kv_blocks);
        int lse_tile_offset =
            (work.batch * params.num_heads + work.head) * params.num_row_tiles + work.row_tile;
        corr_state = mainloop.template correction<SharedStorage, NamedBarriers>(
            params.mainloop.sm_scale_log2, pipeline_s_p_o, pipeline_sm_stats, pipeline_o_acc,
            pipeline_o_epi, shared_storage, tmem_base, tile_nkv, corr_state,
            params.epilogue.ptr_LSE, lse_tile_offset);
        work = tile_sched.consumer_advance();
      }
    } else if (warp_idx >= 4) {
      // ===== WG1 (warps 4-7): Softmax stage=1 =====
      warpgroup_reg_set<kRegsSoftmax>();
      while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {
      }
      __threadfence_block();

      const uint32_t tmem_base = shared_storage.tmem_base_ptr;
      typename CollectiveMainloop::SoftmaxState softmax1_state;
      CLCTileScheduler tile_sched(
          pipeline_clc, shared_storage.pipelines.clc_response,
          {params.num_row_tiles, params.num_kv_blocks, params.num_heads, params.batch});

      auto work = tile_sched.initial_work_tile_info();
      while (work.is_valid) {
        int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
            params.mainloop, work.batch, work.head, work.row_tile, global_num_kv_blocks);
        int const* tile_bi = nullptr;
        if (params.mainloop.ptr_block_indices != nullptr) {
          int tile_idx =
              (work.batch * params.num_heads + work.head) * params.num_row_tiles + work.row_tile;
          tile_bi =
              params.mainloop.ptr_block_indices + tile_idx * params.mainloop.block_indices_stride;
        }
        softmax1_state = mainloop.template softmax</*Stage=*/1, SharedStorage, NamedBarriers>(
            pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit, shared_storage, tmem_base,
            params.mainloop.sm_scale_log2, tile_nkv, softmax1_state, tile_bi,
            params.mainloop.ptr_block_sizes,
            CollectiveMainloop::get_tile_raw_block_count(params.mainloop, work.batch, work.head,
                                                         work.row_tile));
        work = tile_sched.consumer_advance();
      }
    } else {
      // ===== WG0 (warps 0-3): Softmax stage=0 =====
      warpgroup_reg_set<kRegsSoftmax>();
      while (!*reinterpret_cast<volatile int*>(&shared_storage.tmem_ready)) {
      }
      __threadfence_block();

      const uint32_t tmem_base = shared_storage.tmem_base_ptr;
      typename CollectiveMainloop::SoftmaxState softmax0_state;
      CLCTileScheduler tile_sched(
          pipeline_clc, shared_storage.pipelines.clc_response,
          {params.num_row_tiles, params.num_kv_blocks, params.num_heads, params.batch});

      auto work = tile_sched.initial_work_tile_info();
      while (work.is_valid) {
        int tile_nkv = CollectiveMainloop::get_tile_num_kv_blocks(
            params.mainloop, work.batch, work.head, work.row_tile, global_num_kv_blocks);
        int const* tile_bi = nullptr;
        if (params.mainloop.ptr_block_indices != nullptr) {
          int tile_idx =
              (work.batch * params.num_heads + work.head) * params.num_row_tiles + work.row_tile;
          tile_bi =
              params.mainloop.ptr_block_indices + tile_idx * params.mainloop.block_indices_stride;
        }
        softmax0_state = mainloop.template softmax</*Stage=*/0, SharedStorage, NamedBarriers>(
            pipeline_s_p_o, pipeline_sm_stats, pipeline_p_lastsplit, shared_storage, tmem_base,
            params.mainloop.sm_scale_log2, tile_nkv, softmax0_state, tile_bi,
            params.mainloop.ptr_block_sizes,
            CollectiveMainloop::get_tile_raw_block_count(params.mainloop, work.batch, work.head,
                                                         work.row_tile));
        work = tile_sched.consumer_advance();
      }
    }
  }
};

// ---- Free function: device kernel entry point ----
template <class Kernel>
__global__ static void __launch_bounds__(Kernel::kThreads, 1)
    fused_attn_device(__grid_constant__ const typename Kernel::Params params) {
  extern __shared__ char shared_memory[];
  Kernel kernel;
  kernel(params, shared_memory);
}

// ---- Convenience type aliases ----
template <bool HasVariableBlockNums = false, bool HasBlockSizes = true>
using FusedAttnKernel =
    FusedAttnFwdSm100<CollectiveMainloopFwd<HasVariableBlockNums, HasBlockSizes>,
                      CollectiveEpilogueFwd>;

#endif
}  // namespace flash
