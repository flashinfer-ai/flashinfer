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

#include "cute_sm120_mxfp8_groupwise/sm120_common/ab_tma_load.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/epilogue.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/math.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/scheduler.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaled/sf_mxfp8_tma_load.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaled {

template <typename KT>
struct SM120BlockScaledGemmKernel {
  static constexpr int kNumTMAThreads = 128;
  static constexpr int kNumMathThreads = KT::MMAConfig::kNumMathThreads;
  static constexpr int MaxThreadsPerBlock = kNumTMAThreads + kNumMathThreads;
  static constexpr int MinBlocksPerMultiprocessor = 1;

  static constexpr sm120_common::GemmType kGemmType = KT::kGemmType;
  using Scheduler =
      std::conditional_t<KT::kSwapAB, sm120_common::Scheduler<kGemmType, KT::kTileN, KT::kTileM>,
                         sm120_common::Scheduler<kGemmType, KT::kTileM, KT::kTileN>>;
  using ProblemShape = typename KT::ProblemShape;

  struct Params {
    typename KT::ABLoadConfig::TMA_A tma_load_a;
    typename KT::ABLoadConfig::TMA_B tma_load_b;
    typename KT::SFConfig::TMA_SFA tma_load_sfa;
    typename KT::SFConfig::TMA_SFB tma_load_sfb;
    typename KT::TmaStoreConfig::TMA_D tma_store_d;
    typename KT::ElementD* ptr_D;
    int M;
    int N;
    int K;
    int num_experts;
    int32_t const* grouped_layout;
  };

  struct Arguments {
    typename KT::ElementA* ptr_A;
    typename KT::ABLoadConfig::StrideA dA;
    typename KT::ElementB* ptr_B;
    typename KT::ABLoadConfig::StrideB dB;
    typename KT::SFConfig::ElementSFLoad* ptr_SFA;
    typename KT::SFConfig::ElementSFLoad* ptr_SFB;
    typename KT::ElementD* ptr_D;
    int32_t const* grouped_layout;
    typename KT::TmaStoreConfig::StrideD dD = {};
  };

  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, num_experts] = problem_shape;
    auto [tma_load_a, tma_load_b] = sm120_common::utils::make_ab_tma_descriptors<KT>(
        args.ptr_A, args.dA, args.ptr_B, args.dB, M, N, K, num_experts);
    auto [tma_load_sfa, tma_load_sfb] =
        utils::make_sf_tma_descriptors<KT>(args.ptr_SFA, args.ptr_SFB, M, N, K, num_experts);

    typename KT::TmaStoreConfig::TMA_D tma_store_d{};
    if constexpr (KT::kUseTmaStore) {
      auto tensor_d =
          make_tensor(make_gmem_ptr(args.ptr_D),
                      sm120_common::utils::deduce_d_layout<KT::kSwapAB>(M, N, num_experts));
      tma_store_d = make_tma_copy_C_sm90(typename KT::TmaStoreConfig::CopyOpS2G{}, tensor_d,
                                         take<0, 2>(typename KT::TmaStoreConfig::SmemLayoutD{}),
                                         typename KT::TmaStoreConfig::EpilogueTile_MN{});
    }

    return {tma_load_a, tma_load_b, tma_load_sfa, tma_load_sfb,       tma_store_d, args.ptr_D, M,
            N,          K,          num_experts,  args.grouped_layout};
  }

  static dim3 get_grid_shape(int num_sms) { return dim3(num_sms, 1, 1); }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTE_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
  }

  using TensorStorage = std::conditional_t<KT::kUseTmaStore, typename KT::TensorStorageSplit,
                                           typename KT::TensorStorageUnion>;
  using BarrierStorage = typename KT::BarrierStorage;
  struct SharedStorage {
    TensorStorage tensors;
    alignas(16) BarrierStorage barriers;
  };
  static constexpr int kSmemSize = int(sizeof(SharedStorage));

  using FullBarrier = typename KT::FullBarrier;
  using EmptyBarrier = typename KT::EmptyBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  using ConsumerBarrierType = typename EmptyBarrier::ValueType;

  CUTE_DEVICE
  static auto get_mbarriers(SharedStorage& shared_storage) {
    auto* ab_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.ab_full_mbar[0]);
    auto* ab_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.ab_empty_mbar[0]);
    auto* sf_full_mbar = recast_ptr<FullBarrier>(&shared_storage.barriers.sf_full_mbar[0]);
    auto* sf_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.sf_empty_mbar[0]);
    auto* store_full_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.store_full_mbar[0]);
    auto* store_empty_mbar = recast_ptr<EmptyBarrier>(&shared_storage.barriers.store_empty_mbar[0]);
    return cute::make_tuple(ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar,
                            store_full_mbar, store_empty_mbar);
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_sf(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset,
                                  int32_t num_sf_cycles, uint32_t& sf_phase,
                                  uint32_t& store_phase) {
    auto [tAgSFA, tBgSFB] =
        utils::tma_sf_partition<KT>(params.tma_load_sfa, params.tma_load_sfb, params.M, params.N,
                                    params.K, params.num_experts, blk_coord, m_offset);

    auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
    auto block_tma_sfb = params.tma_load_sfb.get_slice(0);

    auto sSFA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                             typename KT::SFConfig::SmemLayoutSFA{});
    auto sSFB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                             typename KT::SFConfig::SmemLayoutSFB{});
    auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
    auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

    auto tAsSFA = block_tma_sfa.partition_D(sSFA);
    auto tBsSFB = block_tma_sfb.partition_D(sSFB);

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = get_mbarriers(shared_storage);
    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    for (int32_t sf_cycle = 0; sf_cycle < num_sf_cycles; ++sf_cycle) {
      cute::for_each(cute::make_int_sequence<KT::SFConfig::SF_Stages>{}, [&](auto sf_stage) {
        sf_empty_mbar[sf_stage].wait(sf_phase);
        auto& sf_full_barrier = sf_full_mbar[sf_stage];
        auto tma_copy_sfa =
            params.tma_load_sfa.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
        cute::copy(tma_copy_sfa, tAgSFA(_, _, _, sf_cycle * KT::SFConfig::SF_Stages + sf_stage),
                   tAsSFA(_, _, _, sf_stage));
        auto tma_copy_sfb =
            params.tma_load_sfb.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
        cute::copy(tma_copy_sfb, tBgSFB(_, _, _, sf_cycle * KT::SFConfig::SF_Stages + sf_stage),
                   tBsSFB(_, _, _, sf_stage));
        sf_full_mbar[sf_stage].arrive_and_expect_tx(KT::SFConfig::TmaSFTransactionBytes);
      });
      sf_phase ^= 1;
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_ab(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset,
                                  int32_t num_sf_cycles, uint32_t& ab_phase,
                                  uint32_t& store_phase) {
    auto [tAgA, tBgB] = sm120_common::utils::tma_ab_partition<KT>(
        params.tma_load_a, params.tma_load_b, params.M, params.N, params.K, params.num_experts,
        blk_coord, m_offset);

    auto block_tma_a = params.tma_load_a.get_slice(0);
    auto block_tma_b = params.tma_load_b.get_slice(0);

    auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
                           typename KT::ABLoadConfig::SmemLayoutA{});
    auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()),
                           typename KT::ABLoadConfig::SmemLayoutB{});
    auto sA = as_position_independent_swizzle_tensor(sA_);
    auto sB = as_position_independent_swizzle_tensor(sB_);

    auto tAsA = block_tma_a.partition_D(sA);
    auto tBsB = block_tma_b.partition_D(sB);

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = get_mbarriers(shared_storage);
    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    int32_t k_tile_count =
        num_sf_cycles * KT::SFConfig::SF_Stages * KT::SFConfig::kNumTileKPerPackSF;
    for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx += KT::AB_Stages) {
      cute::for_each(cute::make_int_sequence<KT::AB_Stages>{}, [&](auto write_stage) {
        ab_empty_mbar[write_stage].wait(ab_phase);
        auto& ab_full_barrier = ab_full_mbar[write_stage];
        auto tma_copy_a =
            params.tma_load_a.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
        cute::copy(tma_copy_a, tAgA(_, _, _, k_tile_idx + write_stage), tAsA(_, _, _, write_stage));
        auto tma_copy_b =
            params.tma_load_b.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
        cute::copy(tma_copy_b, tBgB(_, _, _, k_tile_idx + write_stage), tBsB(_, _, _, write_stage));
        ab_full_mbar[write_stage].arrive_and_expect_tx(KT::ABLoadConfig::TmaABTransactionBytes);
      });
      ab_phase ^= 1;
    }
  }

  CUTE_DEVICE
  static void mma(Params const& params, SharedStorage& shared_storage, int32_t num_sf_cycles,
                  int32_t m_offset, int32_t m_boundary, int32_t m_block_idx, int32_t n_block_idx,
                  int32_t expert_idx, uint32_t& sf_phase, uint32_t& ab_phase, int& epi_stage,
                  uint32_t* se_phase) {
    int thread_idx = int(threadIdx.x);

    auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
                           typename KT::ABLoadConfig::SmemLayoutA{});
    auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()),
                           typename KT::ABLoadConfig::SmemLayoutB{});
    auto sSFA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                             typename KT::SFConfig::SmemLayoutSFA{});
    auto sSFB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                             typename KT::SFConfig::SmemLayoutSFB{});
    auto sA = as_position_independent_swizzle_tensor(sA_);
    auto sB = as_position_independent_swizzle_tensor(sB_);
    auto sSFA = as_position_independent_swizzle_tensor(sSFA_);
    auto sSFB = as_position_independent_swizzle_tensor(sSFB_);

    typename KT::MMAConfig::TiledMma mma;
    auto tile_shape_mnk = tile_shape(mma);
    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto accum = partition_fragment_C(mma, cute::take<0, 2>(typename KT::TileShape{}));
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));
    constexpr int K_BLOCK_MAX = decltype(cute::size<2>(tCrA))::value;

    auto s2r_copy_A = make_tiled_copy_A(typename KT::ABLoadConfig::SmemCopyAtomA{}, mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
    auto tXsA = s2r_thr_copy_A.partition_S(sA);
    auto tXrA = s2r_thr_copy_A.retile_D(tCrA);
    auto s2r_copy_B = make_tiled_copy_B(typename KT::ABLoadConfig::SmemCopyAtomB{}, mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
    auto tXsB = s2r_thr_copy_B.partition_S(sB);
    auto tXrB = s2r_thr_copy_B.retile_D(tCrB);

    auto s2r_copy_SFA = make_tiled_copy_impl(typename KT::SFConfig::SmemCopyAtomSF{},
                                             KT::SFConfig::get_layoutSFA_TV(mma),
                                             make_shape(size<0>(tile_shape(mma)), _1{}));
    auto s2r_thr_copy_SFA = s2r_copy_SFA.get_thread_slice(thread_idx);
    auto tXsSFA = s2r_thr_copy_SFA.partition_S(sSFA);
    auto tCrSFA = KT::SFConfig::partition_fragment_SFA(sSFA(_, _, Int<0>{}), thr_mma);
    auto tXrSFA = s2r_thr_copy_SFA.retile_D(tCrSFA);
    auto tCrSFA_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFA);

    auto s2r_copy_SFB = make_tiled_copy_impl(typename KT::SFConfig::SmemCopyAtomSF{},
                                             KT::SFConfig::get_layoutSFB_TV(mma),
                                             make_shape(size<1>(tile_shape(mma)), _1{}));
    auto s2r_thr_copy_SFB = s2r_copy_SFB.get_thread_slice(thread_idx);
    auto tXsSFB = s2r_thr_copy_SFB.partition_S(sSFB);
    auto tCrSFB = KT::SFConfig::partition_fragment_SFB(sSFB(_, _, Int<0>{}), thr_mma);
    auto tXrSFB = s2r_thr_copy_SFB.retile_D(tCrSFB);
    auto tCrSFB_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFB);

    cute::clear(accum);
    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = get_mbarriers(shared_storage);

    for (int32_t sf_cycle = 0; sf_cycle < num_sf_cycles - 1; ++sf_cycle) {
      cute::for_each(cute::make_int_sequence<KT::SFConfig::SF_Stages>{}, [&](auto sf_stage) {
        sf_full_mbar[sf_stage].wait(sf_phase);
        cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, sf_stage), tXrSFA);
        cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, sf_stage), tXrSFB);
        sf_empty_mbar[sf_stage].arrive();

        cute::for_each(
            cute::make_int_sequence<KT::SFConfig::kNumTileKPerPackSF>{}, [&](auto k_in_sf) {
              constexpr int stage = sf_stage * KT::SFConfig::kNumTileKPerPackSF + k_in_sf;
              constexpr int ab_stage = stage & (KT::AB_Stages - 1);

              ab_full_mbar[ab_stage].wait(ab_phase);
              cute::copy(s2r_copy_A, tXsA(_, _, _, Int<ab_stage>{}), tXrA);
              cute::copy(s2r_copy_B, tXsB(_, _, _, Int<ab_stage>{}), tXrB);
              ab_empty_mbar[ab_stage].arrive();
              if constexpr (ab_stage == KT::AB_Stages - 1) {
                ab_phase ^= 1;
              }

              cute::gemm(mma, make_zip_tensor(tCrA, tCrSFA_frg(_, _, _, k_in_sf)),
                         make_zip_tensor(tCrB, tCrSFB_frg(_, _, _, k_in_sf)), accum);
            });
      });
      sf_phase ^= 1;
    }

    cute::for_each(cute::make_int_sequence<KT::SFConfig::SF_Stages>{}, [&](auto sf_stage) {
      sf_full_mbar[sf_stage].wait(sf_phase);
      cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, sf_stage), tXrSFA);
      cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, sf_stage), tXrSFB);
      sf_empty_mbar[sf_stage].arrive();

      cute::for_each(cute::make_int_sequence<KT::SFConfig::kNumTileKPerPackSF>{},
                     [&](auto k_in_sf) {
                       constexpr int stage = sf_stage * KT::SFConfig::kNumTileKPerPackSF + k_in_sf;
                       constexpr int ab_stage = stage & (KT::AB_Stages - 1);
                       constexpr bool is_last_k_tile =
                           stage == KT::SFConfig::SF_Stages * KT::SFConfig::kNumTileKPerPackSF - 1;

                       ab_full_mbar[ab_stage].wait(ab_phase);
                       cute::copy(s2r_copy_A, tXsA(_, _, _, Int<ab_stage>{}), tXrA);
                       cute::copy(s2r_copy_B, tXsB(_, _, _, Int<ab_stage>{}), tXrB);
                       ab_empty_mbar[ab_stage].arrive();
                       if constexpr (ab_stage == KT::AB_Stages - 1) {
                         ab_phase ^= 1;
                       }

                       if constexpr (KT::kUnionSmem && is_last_k_tile) {
                         cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
                       }

                       cute::gemm(mma, make_zip_tensor(tCrA, tCrSFA_frg(_, _, _, k_in_sf)),
                                  make_zip_tensor(tCrB, tCrSFB_frg(_, _, _, k_in_sf)), accum);
                     });
    });
    sf_phase ^= 1;

    // Epilogue dispatch: kFlat (Normal/Batched/Masked) → TMA store; non-flat MoE:
    //   SwapAB MoE (!kFlat && kSwapAB) → direct STG
    //   non-SwapAB MoE                 → smem+R2G with smem_O union
    if constexpr (KT::kUseTmaStore) {
      sm120_common::utils::epi_r2s<KT>(params, shared_storage, accum, thread_idx, epi_stage,
                                       se_phase, store_full_mbar, store_empty_mbar);
    } else if constexpr (!KT::kFlat && KT::kSwapAB) {
      sm120_common::utils::epi_pred_stg<KT>(params, accum, thread_idx, m_offset, m_boundary,
                                            m_block_idx, n_block_idx, store_empty_mbar);
    } else {
      sm120_common::utils::epi_pred_r2g<KT>(params, shared_storage, accum, thread_idx, m_offset,
                                            m_boundary, m_block_idx, n_block_idx, expert_idx,
                                            store_empty_mbar);
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void store(Params const& params, SharedStorage& shared_storage,
                                BlkCoord const& blk_coord, uint32_t* sf_phase, int& epi_stage) {
    if constexpr (KT::kUseTmaStore) {
      auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
            store_empty_mbar] = get_mbarriers(shared_storage);
      sm120_common::utils::tma_store<KT>(params, shared_storage, cute::get<0>(blk_coord),
                                         cute::get<1>(blk_coord), cute::get<2>(blk_coord), sf_phase,
                                         epi_stage, store_full_mbar, store_empty_mbar);
    }
  }

  CUTE_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_predicate = cute::elect_one_sync();
    bool is_tma_thread = warp_idx == 0 && lane_predicate;

    if (is_tma_thread) {
      prefetch_tma_descriptors(params);
    }
    __syncthreads();

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = get_mbarriers(shared_storage);
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
      if constexpr (KT::kUseTmaStore) {
#pragma unroll
        for (uint32_t i = 0; i < KT::TmaStoreConfig::StagesD; ++i) {
          store_full_mbar[i].init(KT::MMAConfig::kNumMathThreads);
          store_empty_mbar[i].init(1);
        }
      } else if constexpr (KT::kUnionSmem) {
        store_empty_mbar[0].init(KT::MMAConfig::kNumMathThreads);
      }
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    int32_t num_sf_cycles = sm120_common::math::ceil_div(
        params.K, int(KT::SFConfig::PACK_NK * KT::SFConfig::SF_Stages));

    if (warp_idx >= KT::MMAConfig::kNumMathWarps) {
      constexpr int epi_warp_idx = KT::MMAConfig::kNumMathWarps;
      constexpr int ab_warp_idx = epi_warp_idx + 1;
      constexpr int sf_warp_idx = ab_warp_idx + 1;

      if (warp_idx == epi_warp_idx) {
        if constexpr (KT::kUseTmaStore) {
          uint32_t sf_phase[KT::TmaStoreConfig::StagesD] = {0};
          int epi_stage = 0;
          if (lane_predicate) {
            Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
            int32_t m_block_idx, n_block_idx;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
              auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
                  m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx));
              store(params, shared_storage, blk_coord, sf_phase, epi_stage);
            }
          }
          __syncwarp();
        }
      }
      if (warp_idx == ab_warp_idx) {
        uint32_t ab_phase = 1;
        uint32_t store_phase = 1;
        if (lane_predicate) {
          Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
          int32_t m_block_idx, n_block_idx;
          while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
                m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx));
            load_ab(params, shared_storage, blk_coord, scheduler.get_m_offset(), num_sf_cycles,
                    ab_phase, store_phase);
          }
        }
        __syncwarp();
      }
      if (warp_idx == sf_warp_idx) {
        uint32_t sf_phase = 1;
        uint32_t store_phase = 1;
        if (lane_predicate) {
          Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
          int32_t m_block_idx, n_block_idx;
          while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
                m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx));
            load_sf(params, shared_storage, blk_coord, scheduler.get_m_offset(), num_sf_cycles,
                    sf_phase, store_phase);
          }
        }
        __syncwarp();
      }
    } else {
      uint32_t sf_phase = 0;
      uint32_t ab_phase = 0;
      int epi_stage = 0;
      uint32_t se_phase[KT::TmaStoreConfig::StagesD];
#pragma unroll
      for (int i = 0; i < KT::TmaStoreConfig::StagesD; ++i) se_phase[i] = 1;
      Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
      int32_t m_block_idx, n_block_idx;
      while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
            m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx));
        mma(params, shared_storage, num_sf_cycles, scheduler.get_m_offset(),
            scheduler.get_m_boundary(), cute::get<0>(blk_coord), cute::get<1>(blk_coord),
            cute::get<2>(blk_coord), sf_phase, ab_phase, epi_stage, se_phase);
      }
    }
  }
};

}  // namespace sm120_blockscaled
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
