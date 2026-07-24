/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// clang-format off
#include <type_traits>

#include <cuda_runtime.h>

#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>

#include "cute_sm120_mxfp8_groupwise/sm120_blockscaled/kernel_impl.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/moe_scheduler.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaled {

template <typename KT>
struct SM120BlockScaledMoeGemmKernel : SM120BlockScaledGemmKernel<KT> {
  using BaseKernel = SM120BlockScaledGemmKernel<KT>;
  using Params = typename BaseKernel::Params;
  using Scheduler = sm120_common::SelectedMoeScheduler<KT::kSwapAB, KT::kTileM, KT::kTileN>;

  static_assert(KT::kGemmType == sm120_common::GemmType::MGroupedContiguousWithZeroPadding);
  static_assert(KT::kUnionSmem,
                "the dedicated scheduler warp reuses the idle epi warp; a "
                "TmaStore instance needs an explicit scheduling decision");

  static constexpr int kNumSchedStages = 2;
  static constexpr int kNumSchedConsumers = KT::MMAConfig::kNumMathWarps + 2;

  struct SharedStorage : BaseKernel::SharedStorage {
    alignas(8) sm120_common::MoeSchedStorage<kNumSchedStages> sched;
  };
  static constexpr int kSmemSize = int(sizeof(SharedStorage));

  CUTE_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    bool is_tma_thread = warp_idx == 0 && lane_predicate;

    if (is_tma_thread) {
      BaseKernel::prefetch_tma_descriptors(params);
    }
    __syncthreads();

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = BaseKernel::get_mbarriers(shared_storage);
    (void)store_full_mbar;
    if (is_tma_thread) {
#pragma unroll
      for (uint32_t i = 0; i < KT::SFConfig::SF_Stages; ++i) {
        sf_full_mbar[i].init(1);
        sf_empty_mbar[i].init(KT::MMAConfig::kNumMathThreads);
      }
#pragma unroll
      for (uint32_t i = 0; i < KT::AB_Stages; ++i) {
        ab_full_mbar[i].init(1);
        ab_empty_mbar[i].init(KT::MMAConfig::kNumMathThreads);
      }
      store_empty_mbar[0].init(KT::MMAConfig::kNumMathThreads);
      shared_storage.sched.init_mbars(kNumSchedConsumers);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    int32_t num_sf_cycles = sm120_common::math::ceil_div(
        params.K, int(KT::SFConfig::PACK_NK * KT::SFConfig::SF_Stages));

    if (warp_idx >= KT::MMAConfig::kNumMathWarps) {
      constexpr int sched_warp_idx = KT::MMAConfig::kNumMathWarps;
      constexpr int ab_warp_idx = sched_warp_idx + 1;
      constexpr int sf_warp_idx = ab_warp_idx + 1;

      if (warp_idx == sched_warp_idx) {
        Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
        sm120_common::MoeSchedProducer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                       lane_predicate};
        int32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
          sched_pipeline.publish(sm120_common::MoeWorkTile{
              m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx),
              scheduler.get_m_offset(), scheduler.get_m_boundary(), 1});
        }
        sched_pipeline.publish_sentinel();
      }
      if (warp_idx == ab_warp_idx) {
        uint32_t ab_phase = 1;
        uint32_t store_phase = 1;
        sm120_common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                       lane_predicate};
        sm120_common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          if (lane_predicate) {
            auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
                tile.m_block, tile.n_block, tile.group);
            BaseKernel::load_ab(params, shared_storage, blk_coord, tile.m_offset, num_sf_cycles,
                                ab_phase, store_phase);
          }
        }
      }
      if (warp_idx == sf_warp_idx) {
        uint32_t sf_phase = 1;
        uint32_t store_phase = 1;
        sm120_common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                       lane_predicate};
        sm120_common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          if (lane_predicate) {
            auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
                tile.m_block, tile.n_block, tile.group);
            BaseKernel::load_sf(params, shared_storage, blk_coord, tile.m_offset, num_sf_cycles,
                                sf_phase, store_phase);
          }
        }
      }
    } else {
      uint32_t sf_phase = 0;
      uint32_t ab_phase = 0;
      int epi_stage = 0;
      uint32_t se_phase[1] = {1};
      sm120_common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                     lane_predicate};
      sm120_common::MoeWorkTile tile;
      while (sched_pipeline.get_next_tile(tile)) {
        auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(tile.m_block,
                                                                          tile.n_block, tile.group);
        BaseKernel::mma(params, shared_storage, num_sf_cycles, tile.m_offset, tile.m_boundary,
                        cute::get<0>(blk_coord), cute::get<1>(blk_coord), cute::get<2>(blk_coord),
                        sf_phase, ab_phase, epi_stage, se_phase);
      }
    }
  }
};

}  // namespace sm120_blockscaled
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
