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

#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>

#include "cute_sm120_mxfp8_groupwise/sm120_blockscaling/builder.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/ab_tma_load.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/epilogue.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/math.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/scheduler.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaling {

using namespace cute;

template <typename KT>
struct SM120BlockScalingGemmKernel {
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
    typename KT::SfaTmaLoadConfig::TMA_SFA tma_load_sfa;
    typename KT::ElementScale const* ptr_SFA;
    typename KT::ElementScale const* ptr_SFB;
    typename KT::TmaStoreConfig::TMA_D tma_store_d;
    typename KT::ElementD* ptr_D;
    int M;
    int N;
    int K;
    int num_experts;
    int32_t const* grouped_layout;
  };

  struct Arguments {
    typename KT::ElementA const* ptr_A;
    typename KT::ABLoadConfig::StrideA dA;
    typename KT::ElementB const* ptr_B;
    typename KT::ABLoadConfig::StrideB dB;
    typename KT::ElementScale const* ptr_SFA;
    typename KT::ElementScale const* ptr_SFB;
    typename KT::ElementD* ptr_D;
    int32_t const* grouped_layout;
    typename KT::TmaStoreConfig::StrideD dD = {};
  };

  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, num_experts] = problem_shape;
    auto [tma_load_a, tma_load_b] = sm120_common::utils::make_ab_tma_descriptors<KT>(
        args.ptr_A, args.dA, args.ptr_B, args.dB, M, N, K, num_experts);

    typename KT::SfaTmaLoadConfig::TMA_SFA tma_load_sfa{};
    if constexpr (KT::kUseTmaSFA) {
      tma_load_sfa = utils::make_sfa_tma_descriptor<KT>(args.ptr_SFA, M, N, K, num_experts);
    }

    typename KT::TmaStoreConfig::TMA_D tma_store_d{};
    if constexpr (KT::kUseTmaStore) {
      auto tensor_d =
          make_tensor(make_gmem_ptr(args.ptr_D),
                      sm120_common::utils::deduce_d_layout<KT::kSwapAB>(M, N, num_experts));
      tma_store_d = make_tma_copy_C_sm90(typename KT::TmaStoreConfig::CopyOpS2G{}, tensor_d,
                                         take<0, 2>(typename KT::TmaStoreConfig::SmemLayoutD{}),
                                         typename KT::TmaStoreConfig::EpilogueTile_MN{});
    }

    return {tma_load_a,
            tma_load_b,
            tma_load_sfa,
            args.ptr_SFA,
            args.ptr_SFB,
            tma_store_d,
            args.ptr_D,
            M,
            N,
            K,
            num_experts,
            args.grouped_layout};
  }

  static dim3 get_grid_shape(int num_sms) { return dim3(num_sms, 1, 1); }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTE_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    if constexpr (KT::kUseTmaSFA) {
      cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
    }
    if constexpr (KT::kUseTmaStore) {
      cute::prefetch_tma_descriptor(params.tma_store_d.get_tma_descriptor());
    }
  }

  using TensorStorage =
      std::conditional_t<KT::kUseTmaStore, typename KT::TensorStorageSplit,
                         std::conditional_t<KT::kUseStagedR2G, typename KT::TensorStorageStagedR2G,
                                            typename KT::TensorStorageUnion>>;
  using BarrierStorage = typename KT::BarrierStorage;

  struct SharedStorage {
    TensorStorage tensors;
    alignas(16) BarrierStorage barriers;
  };

  static constexpr int kSmemSize = int(sizeof(SharedStorage));

  using FullBarrier = typename KT::FullBarrier;
  using EmptyBarrier = typename KT::EmptyBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;

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
  CUTE_DEVICE static void load_sf_tma(Params const& params, SharedStorage& shared_storage,
                                      BlkCoord const& blk_coord, int32_t m_offset,
                                      int32_t k_tile_count, int& sf_stage, uint32_t& sf_phase,
                                      uint32_t& store_phase) {
    static_assert(KT::kUseTmaSFA);
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    (void)m_block_idx;
    int lane_idx = cutlass::canonical_lane_idx();
    int lane_predicate = cute::elect_one_sync();

    auto sSFA = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                            typename KT::SFConfig::SmemLayoutTmaSFA{});
    auto sSFB = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                            typename KT::SFConfig::SmemLayoutSFB{});

    auto tAgSFA = utils::tma_sfa_partition<KT>(params.tma_load_sfa, params.M, params.N, params.K,
                                               params.num_experts, blk_coord, m_offset);
    auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
    auto tAsSFA = block_tma_sfa.partition_D(sSFA);

    Tensor mSFB = make_tensor(
        make_gmem_ptr(params.ptr_SFB),
        KT::SFConfig::deduce_sfb_layout(params.M, params.N, params.K, params.num_experts));
    Tensor cSFB = make_identity_tensor(mSFB.shape());
    auto scales_n = get<0>(mSFB.shape());
    Tensor gSFB = local_tile(mSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                             make_coord(n_block_idx, _, expert_idx));
    Tensor coordSFB = local_tile(cSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                                 make_coord(n_block_idx, _, expert_idx));

    TiledCopy scale_copy_b = make_tiled_copy(
        typename KT::SFConfig::SmemCopyAtomSFB{},
        Layout<Shape<Int<KT::SFConfig::kNumScaleCopyThreads>>>{}, Layout<Shape<_1>>{});
    auto thr_scale_copy_b = scale_copy_b.get_slice(lane_idx);
    Tensor tBgSFB = thr_scale_copy_b.partition_S(gSFB);
    Tensor tBcSFB = thr_scale_copy_b.partition_S(coordSFB);
    Tensor tBsSFB = thr_scale_copy_b.partition_D(sSFB);
    Tensor tBpSFB = cute::lazy::transform(tBcSFB(_, _, 0), [&](auto const& coord) {
      return lane_idx < KT::SFConfig::kTileScaleN && get<0>(coord) < scales_n;
    });

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = get_mbarriers(shared_storage);
    (void)ab_full_mbar;
    (void)ab_empty_mbar;
    (void)store_full_mbar;
    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; ++k_tile_idx) {
      sf_empty_mbar[sf_stage].wait(sf_phase);
      auto& sf_full_barrier = sf_full_mbar[sf_stage];
      if (lane_predicate) {
        auto tma_copy_sfa =
            params.tma_load_sfa.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
        sf_full_barrier.arrive_and_expect_tx(KT::SfaTmaLoadConfig::TmaTransactionBytesSFA);
        cute::copy(tma_copy_sfa, tAgSFA(_, _, _, k_tile_idx), tAsSFA(_, _, _, sf_stage));
      }
      __syncwarp();
      copy_if(scale_copy_b, tBpSFB, tBgSFB(_, _, k_tile_idx), tBsSFB(_, _, sf_stage));
      cutlass::arch::cpasync_barrier_arrive_noinc(
          recast_ptr<ProducerBarrierType>(&sf_full_barrier));
      ++sf_stage;
      if (sf_stage == KT::SFConfig::SF_Stages) {
        sf_stage = 0;
        sf_phase ^= 1;
      }
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_sf(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset, int32_t m_boundary,
                                  int32_t k_tile_count, int& sf_stage, uint32_t& sf_phase,
                                  uint32_t& store_phase) {
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    constexpr bool kPerBatchAB = KT::kPerBatchAB;
    constexpr bool kSwapAB = KT::kSwapAB;

    int sfa_src_M = kSwapAB ? params.N : params.M;
    int sfa_src_L = kSwapAB ? params.num_experts : (kPerBatchAB ? params.num_experts : 1);
    int sfb_src_N = kSwapAB ? params.M : params.N;
    int sfb_src_L = (kSwapAB && !kPerBatchAB) ? 1 : params.num_experts;
    int sfa_real_M = sfa_src_M;
    int sfb_real_N = sfb_src_N;
    int32_t sfa_tile_idx = m_block_idx;
    int32_t sfa_batch_idx = kSwapAB ? expert_idx : (kPerBatchAB ? expert_idx : 0);
    int32_t sfb_tile_idx = n_block_idx;
    int32_t sfb_batch_idx = (kSwapAB && !kPerBatchAB) ? 0 : expert_idx;
    int32_t sf_m_offset = m_offset;
    int lane_idx = cutlass::canonical_lane_idx();

    if constexpr (kSwapAB && KT::kFlat) {
      static_assert(KT::kGranN == 1);
      sfb_real_N = KT::SFConfig::get_tma_aligned_size(sfb_src_N);
    }
    if constexpr (kSwapAB &&
                  KT::kGemmType == sm120_common::GemmType::MGroupedContiguousWithZeroPadding) {
      static_assert(KT::kGranN == 1);
      sfb_real_N = sm120_common::math::compute_padded_offset(sfb_src_N, params.num_experts);
      sf_m_offset = sm120_common::math::compute_padded_offset(m_offset, expert_idx);
    }

    auto sSFA = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                            typename KT::SFConfig::SmemLayoutSFA{});
    auto sSFB = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                            typename KT::SFConfig::SmemLayoutSFB{});

    Tensor mSFA_full =
        make_tensor(make_gmem_ptr(params.ptr_SFA),
                    KT::SFConfig::deduce_sfa_layout(sfa_real_M, sfb_real_N, params.K, sfa_src_L));
    Tensor mSFB_full =
        make_tensor(make_gmem_ptr(params.ptr_SFB),
                    KT::SFConfig::deduce_sfb_layout(sfa_real_M, sfb_real_N, params.K, sfb_src_L));
    Tensor cSFA_full = make_identity_tensor(mSFA_full.shape());
    Tensor cSFB_full = make_identity_tensor(mSFB_full.shape());

    auto mSFA = [&] {
      if constexpr (!kSwapAB && !KT::kFlat) {
        return domain_offset(make_coord(m_offset / KT::kGranM, 0, 0), mSFA_full);
      } else {
        return mSFA_full;
      }
    }();
    auto mSFB = [&] {
      if constexpr (kSwapAB && !KT::kFlat) {
        return domain_offset(make_coord(sf_m_offset / KT::kGranN, 0, 0), mSFB_full);
      } else {
        return mSFB_full;
      }
    }();
    auto cSFA = [&] {
      if constexpr (!kSwapAB && !KT::kFlat) {
        return domain_offset(make_coord(m_offset / KT::kGranM, 0, 0), cSFA_full);
      } else {
        return cSFA_full;
      }
    }();
    auto cSFB = [&] {
      if constexpr (kSwapAB && !KT::kFlat) {
        return domain_offset(make_coord(sf_m_offset / KT::kGranN, 0, 0), cSFB_full);
      } else {
        return cSFB_full;
      }
    }();

    int64_t scales_m = sm120_common::math::ceil_div(sfa_src_M, int(KT::kGranM));
    int64_t scales_n = sm120_common::math::ceil_div(sfb_src_N, int(KT::kGranN));
    if constexpr (kSwapAB &&
                  KT::kGemmType == sm120_common::GemmType::MGroupedContiguousWithZeroPadding) {
      scales_n = sf_m_offset / KT::kGranN +
                 sm120_common::math::ceil_div(m_boundary - m_offset, int(KT::kGranN));
    }

    Tensor gSFA = local_tile(mSFA, make_tile(Int<KT::SFConfig::kTileScaleM>{}),
                             make_coord(sfa_tile_idx, _, sfa_batch_idx));
    Tensor coordSFA = local_tile(cSFA, make_tile(Int<KT::SFConfig::kTileScaleM>{}),
                                 make_coord(sfa_tile_idx, _, sfa_batch_idx));
    Tensor gSFB = local_tile(mSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                             make_coord(sfb_tile_idx, _, sfb_batch_idx));
    Tensor coordSFB = local_tile(cSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                                 make_coord(sfb_tile_idx, _, sfb_batch_idx));

    TiledCopy scale_copy_a = make_tiled_copy(
        typename KT::SFConfig::SmemCopyAtomSFA{},
        Layout<Shape<Int<KT::SFConfig::kNumScaleCopyThreads>>>{}, Layout<Shape<_1>>{});
    TiledCopy scale_copy_b = make_tiled_copy(
        typename KT::SFConfig::SmemCopyAtomSFB{},
        Layout<Shape<Int<KT::SFConfig::kNumScaleCopyThreads>>>{}, Layout<Shape<_1>>{});

    auto thr_scale_copy_a = scale_copy_a.get_slice(lane_idx);
    auto thr_scale_copy_b = scale_copy_b.get_slice(lane_idx);

    Tensor tAgSFA = thr_scale_copy_a.partition_S(gSFA);
    Tensor tAcSFA = thr_scale_copy_a.partition_S(coordSFA);
    Tensor tAsSFA = thr_scale_copy_a.partition_D(sSFA);

    Tensor tBgSFB = thr_scale_copy_b.partition_S(gSFB);
    Tensor tBcSFB = thr_scale_copy_b.partition_S(coordSFB);
    Tensor tBsSFB = thr_scale_copy_b.partition_D(sSFB);

    Tensor tApSFA = cute::lazy::transform(tAcSFA(_, _, 0), [&](auto const& coord) {
      return lane_idx < KT::SFConfig::kTileScaleM && get<0>(coord) < scales_m;
    });
    Tensor tBpSFB = cute::lazy::transform(tBcSFB(_, _, 0), [&](auto const& coord) {
      return lane_idx < KT::SFConfig::kTileScaleN && get<0>(coord) < scales_n;
    });

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = get_mbarriers(shared_storage);
    (void)ab_full_mbar;
    (void)ab_empty_mbar;
    (void)store_full_mbar;
    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; ++k_tile_idx) {
      sf_empty_mbar[sf_stage].wait(sf_phase);

      copy_if(scale_copy_a, tApSFA, tAgSFA(_, _, k_tile_idx), tAsSFA(_, _, sf_stage));
      copy_if(scale_copy_b, tBpSFB, tBgSFB(_, _, k_tile_idx), tBsSFB(_, _, sf_stage));
      cutlass::arch::cpasync_barrier_arrive_noinc(
          recast_ptr<ProducerBarrierType>(&sf_full_mbar[sf_stage]));
      ++sf_stage;
      if (sf_stage == KT::SFConfig::SF_Stages) {
        sf_stage = 0;
        sf_phase ^= 1;
      }
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_ab(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset, int32_t k_tile_count,
                                  int& ab_stage, uint32_t& ab_phase, uint32_t& store_phase) {
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
    (void)sf_full_mbar;
    (void)sf_empty_mbar;
    (void)store_full_mbar;
    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; ++k_tile_idx) {
      ab_empty_mbar[ab_stage].wait(ab_phase);
      auto& ab_full_barrier = ab_full_mbar[ab_stage];
      auto tma_copy_a = params.tma_load_a.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
      cute::copy(tma_copy_a, tAgA(_, _, _, k_tile_idx), tAsA(_, _, _, ab_stage));
      auto tma_copy_b = params.tma_load_b.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
      cute::copy(tma_copy_b, tBgB(_, _, _, k_tile_idx), tBsB(_, _, _, ab_stage));
      ab_full_barrier.arrive_and_expect_tx(KT::ABLoadConfig::TmaABTransactionBytes);
      ++ab_stage;
      if (ab_stage == KT::AB_Stages) {
        ab_stage = 0;
        ab_phase ^= 1;
      }
    }
  }

  CUTE_DEVICE
  static void mma(Params const& params, SharedStorage& shared_storage, int32_t k_tile_count,
                  int32_t m_offset, int32_t m_boundary, int32_t m_block_idx, int32_t n_block_idx,
                  int32_t expert_idx, int& read_stage, uint32_t& sf_phase, uint32_t& ab_phase,
                  int& epi_stage, uint32_t* se_phase) {
    int thread_idx = int(threadIdx.x);
    (void)m_offset;
    (void)m_boundary;
    (void)m_block_idx;
    (void)n_block_idx;
    (void)expert_idx;

    typename KT::MMAConfig::TiledMma mma;
    auto accum = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    clear(accum);

    auto sA_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
                           typename KT::ABLoadConfig::SmemLayoutA{});
    auto sB_ = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()),
                           typename KT::ABLoadConfig::SmemLayoutB{});
    auto sA = as_position_independent_swizzle_tensor(sA_);
    auto sB = as_position_independent_swizzle_tensor(sB_);

    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto tmp_accum = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));
    constexpr int K_BLOCK_MAX = decltype(size<2>(tCrA))::value;

    auto s2r_copy_A = make_tiled_copy_A(typename KT::ABLoadConfig::SmemCopyAtomA{}, mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
    auto tXsA = s2r_thr_copy_A.partition_S(sA);
    auto tXrA = s2r_thr_copy_A.retile_D(tCrA);

    auto s2r_copy_B = make_tiled_copy_B(typename KT::ABLoadConfig::SmemCopyAtomB{}, mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
    auto tXsB = s2r_thr_copy_B.partition_S(sB);
    auto tXrB = s2r_thr_copy_B.retile_D(tCrB);

    auto sSFAViewAsC = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                                   typename KT::SFConfig::SmemLayoutSFAViewAsC{});
    auto sSFBViewAsC = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                                   typename KT::SFConfig::SmemLayoutSFBViewAsC{});
    auto tCsSFAViewAsC = thr_mma.partition_C(sSFAViewAsC);
    auto tCsSFBViewAsC = thr_mma.partition_C(sSFBViewAsC);
    auto tCrSFAViewAsC =
        make_tensor_like<typename KT::ElementScale>(tCsSFAViewAsC(_, _, _, Int<0>{}));
    auto tCrSFBViewAsC =
        make_tensor_like<typename KT::ElementScale>(tCsSFBViewAsC(_, _, _, Int<0>{}));

    auto copy_scale_s2r = [&](int read_stage) {
      copy(tCsSFAViewAsC(_, _, _, read_stage), tCrSFAViewAsC);
      copy(tCsSFBViewAsC(_, _, _, read_stage), tCrSFBViewAsC);
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN == 1) {
        tCrSFAViewAsC.data()[0] = tCrSFAViewAsC.data()[0] * tCrSFBViewAsC.data()[0];
      }
      if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN == 1) {
        typename KT::ElementScale scale_b = tCrSFBViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrSFAViewAsC); ++i) {
          tCrSFAViewAsC.data()[i] = tCrSFAViewAsC.data()[i] * scale_b;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN > 1) {
        typename KT::ElementScale scale_a = tCrSFAViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrSFBViewAsC); ++i) {
          tCrSFBViewAsC.data()[i] = tCrSFBViewAsC.data()[i] * scale_a;
        }
      }
    };

    auto rescale = [&]() {
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN == 1) {
        typename KT::ElementScale scale_ab = tCrSFAViewAsC.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          accum(i) += tmp_accum(i) * scale_ab;
          tmp_accum(i) = 0.0f;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN == 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          accum(i) += tmp_accum(i) * tCrSFAViewAsC(i);
          tmp_accum(i) = 0.0f;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          accum(i) += tmp_accum(i) * tCrSFBViewAsC(i);
          tmp_accum(i) = 0.0f;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
          accum(i) += tmp_accum(i) * tCrSFAViewAsC(i) * tCrSFBViewAsC(i);
          tmp_accum(i) = 0.0f;
        }
      }
    };

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = get_mbarriers(shared_storage);

    clear(tmp_accum);

    auto tXsA_stage = tXsA(_, _, _, read_stage);
    auto tXsB_stage = tXsB(_, _, _, read_stage);
    auto copy_ab_s2r = [&](auto k_block) {
      copy(s2r_copy_A, tXsA_stage(_, _, k_block), tXrA(_, _, k_block));
      copy(s2r_copy_B, tXsB_stage(_, _, k_block), tXrB(_, _, k_block));
    };

    auto advance_read_stage = [&]() {
      ++read_stage;
      if (read_stage == KT::AB_Stages) {
        read_stage = 0;
        sf_phase ^= 1;
        ab_phase ^= 1;
      }
    };

    sf_full_mbar[read_stage].wait(sf_phase);
    copy_scale_s2r(read_stage);
    sf_empty_mbar[read_stage].arrive();

    ab_full_mbar[read_stage].wait(ab_phase);
    copy_ab_s2r(Int<0>{});

    for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count - 1; ++k_tile_idx) {
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block == K_BLOCK_MAX - 1) {
          ab_empty_mbar[read_stage].arrive();
          advance_read_stage();
          tXsA_stage = tXsA(_, _, _, read_stage);
          tXsB_stage = tXsB(_, _, _, read_stage);
          ab_full_mbar[read_stage].wait(ab_phase);
        }
        copy_ab_s2r(k_block_next);
        cute::gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
        if (k_block == K_BLOCK_MAX - 1) {
          rescale();
          sf_full_mbar[read_stage].wait(sf_phase);
          copy_scale_s2r(read_stage);
          sf_empty_mbar[read_stage].arrive();
        }
      });
    }

    cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
      if (k_block_next > 0) {
        copy_ab_s2r(k_block_next);
      }
      if (k_block == K_BLOCK_MAX - 1) {
        ab_empty_mbar[read_stage].arrive();
        if constexpr (KT::kUnionSmem) {
          cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
        }
        advance_read_stage();
      }
      cute::gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
    });
    rescale();

    if constexpr (KT::kUseTmaStore) {
      sm120_common::utils::epi_r2s<KT>(params, shared_storage, accum, thread_idx, epi_stage,
                                       se_phase, store_full_mbar, store_empty_mbar);
    } else if constexpr (KT::kUseStagedR2G) {
      sm120_common::utils::epi_staged_r2s<KT>(params, shared_storage, accum, thread_idx, m_offset,
                                              m_boundary, m_block_idx, epi_stage, se_phase[0],
                                              store_full_mbar, store_empty_mbar);
    } else if constexpr (KT::kFlat || (!KT::kFlat && KT::kSwapAB)) {
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
      (void)ab_full_mbar;
      (void)ab_empty_mbar;
      (void)sf_full_mbar;
      (void)sf_empty_mbar;
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
        sf_full_mbar[i].init(KT::kNumProducerThreadEvents);
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
      } else if constexpr (KT::kUseStagedR2G) {
#pragma unroll
        for (uint32_t i = 0; i < KT::StagedR2GStoreConfig::StagesD; ++i) {
          store_full_mbar[i].init(KT::MMAConfig::kNumMathThreads);
          store_empty_mbar[i].init(KT::StagedR2GStoreConfig::kNumStoreThreads);
        }
      } else if constexpr (KT::kUnionSmem) {
        store_empty_mbar[0].init(KT::MMAConfig::kNumMathThreads);
      }
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    int32_t k_tile_count = sm120_common::math::ceil_div(params.K, int(KT::kTileK));

    if (warp_idx >= KT::MMAConfig::kNumMathWarps) {
      cutlass::arch::warpgroup_reg_dealloc<KT::LoadRegisterRequirement>();
      constexpr int first_specialized_warp_idx = KT::MMAConfig::kNumMathWarps;
      constexpr int tma_store_warp_idx = first_specialized_warp_idx;
      constexpr int ab_warp_idx = first_specialized_warp_idx + 1;
      constexpr int sf_warp_idx = first_specialized_warp_idx + 2;

      if (warp_idx == ab_warp_idx) {
        int ab_stage = 0;
        uint32_t ab_phase = 1;
        uint32_t store_phase = 1;
        if (lane_predicate) {
          Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
          int32_t m_block_idx, n_block_idx;
          while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
                m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx));
            load_ab(params, shared_storage, blk_coord, scheduler.get_m_offset(), k_tile_count,
                    ab_stage, ab_phase, store_phase);
          }
        }
        __syncwarp();
      } else if (warp_idx == sf_warp_idx) {
        int sf_stage = 0;
        uint32_t sf_phase = 1;
        uint32_t store_phase = 1;
        Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
        int32_t m_block_idx, n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
          auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
              m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx));
          if constexpr (KT::kUseTmaSFA) {
            load_sf_tma(params, shared_storage, blk_coord, scheduler.get_m_offset(), k_tile_count,
                        sf_stage, sf_phase, store_phase);
          } else {
            load_sf(params, shared_storage, blk_coord, scheduler.get_m_offset(),
                    scheduler.get_m_boundary(), k_tile_count, sf_stage, sf_phase, store_phase);
          }
        }
        __syncwarp();
      } else {
        if constexpr (KT::kUseTmaStore) {
          if (warp_idx == tma_store_warp_idx) {
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
        } else if constexpr (KT::kUseStagedR2G) {
          if (warp_idx == tma_store_warp_idx) {
            uint32_t full_phase = 0;
            int store_stage = 0;
            int store_thread_idx = cutlass::canonical_lane_idx();
            Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
            int32_t m_block_idx, n_block_idx;
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
              auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
                  m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx));
              sm120_common::utils::staged_r2g_store<KT>(
                  params, shared_storage, scheduler.get_m_offset(), scheduler.get_m_boundary(),
                  cute::get<0>(blk_coord), cute::get<1>(blk_coord), store_thread_idx, full_phase,
                  store_stage, store_full_mbar, store_empty_mbar);
            }
            __syncwarp();
          }
        }
      }
    } else {
      cutlass::arch::warpgroup_reg_alloc<KT::MmaRegisterRequirement>();
      int read_stage = 0;
      uint32_t sf_phase = 0;
      uint32_t ab_phase = 0;
      int epi_stage = 0;
      uint32_t se_phase[KT::TmaStoreConfig::StagesD];
#pragma unroll
      for (int i = 0; i < KT::TmaStoreConfig::StagesD; ++i) {
        se_phase[i] = 1;
      }
      Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
      int32_t m_block_idx, n_block_idx;
      while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
        auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
            m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx));
        mma(params, shared_storage, k_tile_count, scheduler.get_m_offset(),
            scheduler.get_m_boundary(), cute::get<0>(blk_coord), cute::get<1>(blk_coord),
            cute::get<2>(blk_coord), read_stage, sf_phase, ab_phase, epi_stage, se_phase);
      }
    }
  }
};

}  // namespace sm120_blockscaling
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
