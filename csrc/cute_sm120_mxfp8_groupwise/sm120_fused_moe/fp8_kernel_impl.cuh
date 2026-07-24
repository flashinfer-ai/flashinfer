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

#include "cute_sm120_mxfp8_groupwise/sm120_fused_moe/fp8_builder.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaling/kernel_impl.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/moe_scheduler.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaling {

using namespace cute;

template <typename KT>
struct SM120BlockScalingFusedMoeGemmKernel : SM120BlockScalingGemmKernel<KT> {
  using BaseKernel = SM120BlockScalingGemmKernel<KT>;
  using Scheduler = sm120_common::SelectedMoeScheduler<KT::kSwapAB, KT::kTileM, KT::kTileN>;
  using ProblemShape = typename BaseKernel::ProblemShape;
  static constexpr int kNumSchedStages = 2;
  static constexpr int kNumSchedConsumers =
      KT::MMAConfig::kNumMathWarps + (KT::kUseStagedR2G ? 3 : 2);
  using TensorStorage = std::conditional_t<KT::kUseStagedR2G, typename KT::TensorStorageStagedR2G,
                                           typename KT::TensorStorageUnion>;
  struct SharedStorage {
    TensorStorage tensors;
    alignas(16) typename KT::BarrierStorage barriers;
    alignas(8) sm120_common::MoeSchedStorage<kNumSchedStages> sched;
  };
  static constexpr int kSmemSize = int(sizeof(SharedStorage));
  using FullBarrier = typename KT::FullBarrier;
  using EmptyBarrier = typename KT::EmptyBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  static constexpr bool kUseStagedR2G = KT::kUseStagedR2G;
  static constexpr uint32_t TmaTransactionBytesAB =
      KT::kSwapAB
          ? 2 * KT::ABLoadConfig::TmaTransactionBytesA + KT::ABLoadConfig::TmaTransactionBytesB
          : KT::ABLoadConfig::TmaTransactionBytesA + 2 * KT::ABLoadConfig::TmaTransactionBytesB;

  static_assert(KT::kGemmType == sm120_common::GemmType::MGroupedContiguousWithZeroPadding);
  static_assert(KT::SFConfig::SF_Stages == KT::AB_Stages);

  struct Params {
    typename KT::ABLoadConfig::TMA_A tma_load_a;
    typename KT::ABLoadConfig::TMA_B tma_load_b;
    typename KT::SfaTmaLoadConfig::TMA_SFA tma_load_sfa;
    typename KT::ElementScale const* ptr_SFA;
    typename KT::ElementScale const* ptr_SFB;
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
  };

  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, num_experts] = problem_shape;
    auto [tma_load_a, tma_load_b] = sm120_common::utils::make_ab_tma_descriptors<KT>(
        args.ptr_A, args.dA, args.ptr_B, args.dB, M, 2 * N, K, num_experts);
    typename KT::SfaTmaLoadConfig::TMA_SFA tma_load_sfa{};
    if constexpr (KT::kUseTmaSFA) {
      tma_load_sfa = utils::make_sfa_tma_descriptor<KT>(args.ptr_SFA, M, N, K, num_experts);
    }
    return {tma_load_a, tma_load_b, tma_load_sfa, args.ptr_SFA,       args.ptr_SFB, args.ptr_D, M,
            N,          K,          num_experts,  args.grouped_layout};
  }

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

  CUTE_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    if constexpr (KT::kUseTmaSFA) {
      cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_ab(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset, int32_t k_tile_count,
                                  int& ab_stage, uint32_t& ab_phase, uint32_t& store_phase) {
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    using X = Underscore;

    auto mA_full = params.tma_load_a.get_tma_tensor(make_shape(params.M, params.K, 1));
    auto mA = domain_offset(make_coord(m_offset, 0, 0), mA_full);
    auto gA_mkl = local_tile(mA, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    auto gA = gA_mkl(_, _, m_block_idx, _, Int<0>{});

    auto mB =
        params.tma_load_b.get_tma_tensor(make_shape(2 * params.N, params.K, params.num_experts));
    auto mB_up = mB;
    auto mB_gate = domain_offset(make_coord(params.N, 0, 0), mB);
    auto gB_up_nkl =
        local_tile(mB_up, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto gB_gate_nkl =
        local_tile(mB_gate, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto gB_up = gB_up_nkl(_, _, n_block_idx, _, expert_idx);
    auto gB_gate = gB_gate_nkl(_, _, n_block_idx, _, expert_idx);

    auto block_tma_a = params.tma_load_a.get_slice(0);
    auto block_tma_b = params.tma_load_b.get_slice(0);
    auto tAgA = block_tma_a.partition_S(gA);
    auto tBgB_up = block_tma_b.partition_S(gB_up);
    auto tBgB_gate = block_tma_b.partition_S(gB_gate);

    auto sA = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
                    typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sB_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_up.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto sB_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_gate.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto tAsA = block_tma_a.partition_D(sA);
    auto tBsB_up = block_tma_b.partition_D(sB_up);
    auto tBsB_gate = block_tma_b.partition_D(sB_gate);

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
      ab_full_barrier.arrive_and_expect_tx(TmaTransactionBytesAB);
      auto tma_copy_a = params.tma_load_a.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
      auto tma_copy_b = params.tma_load_b.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
      cute::copy(tma_copy_a, tAgA(_, _, _, k_tile_idx), tAsA(_, _, _, ab_stage));
      cute::copy(tma_copy_b, tBgB_up(_, _, _, k_tile_idx), tBsB_up(_, _, _, ab_stage));
      cute::copy(tma_copy_b, tBgB_gate(_, _, _, k_tile_idx), tBsB_gate(_, _, _, ab_stage));
      ++ab_stage;
      if (ab_stage == KT::AB_Stages) {
        ab_stage = 0;
        ab_phase ^= 1;
      }
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_ab_swap(Params const& params, SharedStorage& shared_storage,
                                       BlkCoord const& blk_coord, int32_t m_offset,
                                       int32_t k_tile_count, int& ab_stage, uint32_t& ab_phase,
                                       uint32_t& store_phase) {
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    using X = Underscore;

    auto mA =
        params.tma_load_a.get_tma_tensor(make_shape(2 * params.N, params.K, params.num_experts));
    auto mA_up = mA;
    auto mA_gate = domain_offset(make_coord(params.N, 0, 0), mA);
    auto gA_up_mkl =
        local_tile(mA_up, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    auto gA_gate_mkl =
        local_tile(mA_gate, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
    auto gA_up = gA_up_mkl(_, _, m_block_idx, _, expert_idx);
    auto gA_gate = gA_gate_mkl(_, _, m_block_idx, _, expert_idx);

    auto mB_full = params.tma_load_b.get_tma_tensor(make_shape(params.M, params.K, 1));
    auto mB = domain_offset(make_coord(m_offset, 0, 0), mB_full);
    auto gB_nkl = local_tile(mB, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
    auto gB = gB_nkl(_, _, n_block_idx, _, Int<0>{});

    auto block_tma_a = params.tma_load_a.get_slice(0);
    auto block_tma_b = params.tma_load_b.get_slice(0);
    auto tAgA_up = block_tma_a.partition_S(gA_up);
    auto tAgA_gate = block_tma_a.partition_S(gA_gate);
    auto tBgB = block_tma_b.partition_S(gB);

    auto sA_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A_up.begin()),
                    typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sA_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A_gate.begin()),
                    typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sB = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto tAsA_up = block_tma_a.partition_D(sA_up);
    auto tAsA_gate = block_tma_a.partition_D(sA_gate);
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
      ab_full_barrier.arrive_and_expect_tx(TmaTransactionBytesAB);
      auto tma_copy_a = params.tma_load_a.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
      auto tma_copy_b = params.tma_load_b.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
      cute::copy(tma_copy_a, tAgA_up(_, _, _, k_tile_idx), tAsA_up(_, _, _, ab_stage));
      cute::copy(tma_copy_a, tAgA_gate(_, _, _, k_tile_idx), tAsA_gate(_, _, _, ab_stage));
      cute::copy(tma_copy_b, tBgB(_, _, _, k_tile_idx), tBsB(_, _, _, ab_stage));
      ++ab_stage;
      if (ab_stage == KT::AB_Stages) {
        ab_stage = 0;
        ab_phase ^= 1;
      }
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_sf(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset, int32_t k_tile_count,
                                  int& sf_stage, uint32_t& sf_phase, uint32_t& store_phase) {
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    (void)m_block_idx;
    int lane_idx = cutlass::canonical_lane_idx();
    int lane_predicate = cute::elect_one_sync();

    auto tAgSFA = utils::tma_sfa_partition<KT>(params.tma_load_sfa, params.M, params.N, params.K,
                                               params.num_experts, blk_coord, m_offset);
    auto block_tma_sfa = params.tma_load_sfa.get_slice(0);

    auto sSFA = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                            typename KT::SFConfig::SmemLayoutTmaSFA{});
    auto tAsSFA = block_tma_sfa.partition_D(sSFA);

    auto sfb_layout =
        KT::SFConfig::deduce_sfb_layout(params.M, 2 * params.N, params.K, params.num_experts);
    auto mSFB = make_tensor(make_gmem_ptr(params.ptr_SFB), sfb_layout);
    auto cSFB = make_identity_tensor(mSFB.shape());
    int n_offset = n_block_idx * KT::kTileN;
    int sfb_up_n = n_offset / KT::kGranN;
    int sfb_gate_n = (params.N + n_offset) / KT::kGranN;
    auto gSFB_up = local_tile(mSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                              make_coord(sfb_up_n, _, expert_idx));
    auto gSFB_gate = local_tile(mSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                                make_coord(sfb_gate_n, _, expert_idx));
    auto coordSFB_up = local_tile(cSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                                  make_coord(sfb_up_n, _, expert_idx));
    auto coordSFB_gate = local_tile(cSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                                    make_coord(sfb_gate_n, _, expert_idx));

    auto sSFB_up = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_up.begin()),
                               typename KT::SFConfig::SmemLayoutSFB{});
    auto sSFB_gate = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_gate.begin()),
                                 typename KT::SFConfig::SmemLayoutSFB{});
    TiledCopy scale_copy_b = make_tiled_copy(
        typename KT::SFConfig::SmemCopyAtomSFB{},
        Layout<Shape<Int<KT::SFConfig::kNumScaleCopyThreads>>>{}, Layout<Shape<_1>>{});
    auto thr_scale_copy_b = scale_copy_b.get_slice(lane_idx);
    auto tBgSFB_up = thr_scale_copy_b.partition_S(gSFB_up);
    auto tBgSFB_gate = thr_scale_copy_b.partition_S(gSFB_gate);
    auto tBcSFB_up = thr_scale_copy_b.partition_S(coordSFB_up);
    auto tBcSFB_gate = thr_scale_copy_b.partition_S(coordSFB_gate);
    auto tBsSFB_up = thr_scale_copy_b.partition_D(sSFB_up);
    auto tBsSFB_gate = thr_scale_copy_b.partition_D(sSFB_gate);
    int64_t scales_n = sm120_common::math::ceil_div(2 * params.N, int(KT::kGranN));
    auto tBpSFB_up = cute::lazy::transform(tBcSFB_up(_, _, 0), [&](auto const& coord) {
      return lane_idx < KT::SFConfig::kTileScaleN && get<0>(coord) < scales_n;
    });
    auto tBpSFB_gate = cute::lazy::transform(tBcSFB_gate(_, _, 0), [&](auto const& coord) {
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
        sf_full_barrier.arrive_and_expect_tx(KT::SfaTmaLoadConfig::TmaTransactionBytesSFA);
        auto tma_copy_sfa =
            params.tma_load_sfa.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
        cute::copy(tma_copy_sfa, tAgSFA(_, _, _, k_tile_idx), tAsSFA(_, _, _, sf_stage));
      }
      __syncwarp();
      copy_if(scale_copy_b, tBpSFB_up, tBgSFB_up(_, _, k_tile_idx), tBsSFB_up(_, _, sf_stage));
      copy_if(scale_copy_b, tBpSFB_gate, tBgSFB_gate(_, _, k_tile_idx),
              tBsSFB_gate(_, _, sf_stage));
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
  CUTE_DEVICE static void load_sf_swap(Params const& params, SharedStorage& shared_storage,
                                       BlkCoord const& blk_coord, int32_t m_offset,
                                       int32_t m_boundary, int32_t k_tile_count, int& sf_stage,
                                       uint32_t& sf_phase, uint32_t& store_phase) {
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    int lane_idx = cutlass::canonical_lane_idx();

    auto sfa_layout =
        KT::SFConfig::deduce_sfa_layout(2 * params.N, params.M, params.K, params.num_experts);
    auto mSFA = make_tensor(make_gmem_ptr(params.ptr_SFA), sfa_layout);
    auto cSFA = make_identity_tensor(mSFA.shape());
    auto mSFA_up = mSFA;
    auto mSFA_gate = domain_offset(make_coord(params.N / KT::kGranM, 0, 0), mSFA);
    auto cSFA_up = cSFA;
    auto cSFA_gate = domain_offset(make_coord(params.N / KT::kGranM, 0, 0), cSFA);
    auto gSFA_up = local_tile(mSFA_up, make_tile(Int<KT::SFConfig::kTileScaleM>{}),
                              make_coord(m_block_idx, _, expert_idx));
    auto gSFA_gate = local_tile(mSFA_gate, make_tile(Int<KT::SFConfig::kTileScaleM>{}),
                                make_coord(m_block_idx, _, expert_idx));
    auto coordSFA_up = local_tile(cSFA_up, make_tile(Int<KT::SFConfig::kTileScaleM>{}),
                                  make_coord(m_block_idx, _, expert_idx));
    auto coordSFA_gate = local_tile(cSFA_gate, make_tile(Int<KT::SFConfig::kTileScaleM>{}),
                                    make_coord(m_block_idx, _, expert_idx));

    int32_t sfb_real_N = sm120_common::math::compute_padded_offset(params.M, params.num_experts);
    int32_t sf_m_offset = sm120_common::math::compute_padded_offset(m_offset, expert_idx);
    auto sfb_layout = KT::SFConfig::deduce_sfb_layout(2 * params.N, sfb_real_N, params.K, 1);
    auto mSFB_full = make_tensor(make_gmem_ptr(params.ptr_SFB), sfb_layout);
    auto cSFB_full = make_identity_tensor(mSFB_full.shape());
    auto mSFB = domain_offset(make_coord(sf_m_offset / KT::kGranN, 0, 0), mSFB_full);
    auto cSFB = domain_offset(make_coord(sf_m_offset / KT::kGranN, 0, 0), cSFB_full);
    auto gSFB = local_tile(mSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                           make_coord(n_block_idx, _, Int<0>{}));
    auto coordSFB = local_tile(cSFB, make_tile(Int<KT::SFConfig::kTileScaleN>{}),
                               make_coord(n_block_idx, _, Int<0>{}));

    auto sSFA_up = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA_up.begin()),
                               typename KT::SFConfig::SmemLayoutSFA{});
    auto sSFA_gate = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA_gate.begin()),
                                 typename KT::SFConfig::SmemLayoutSFA{});
    auto sSFB = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                            typename KT::SFConfig::SmemLayoutSFB{});

    TiledCopy scale_copy_a = make_tiled_copy(
        typename KT::SFConfig::SmemCopyAtomSFA{},
        Layout<Shape<Int<KT::SFConfig::kNumScaleCopyThreads>>>{}, Layout<Shape<_1>>{});
    TiledCopy scale_copy_b = make_tiled_copy(
        typename KT::SFConfig::SmemCopyAtomSFB{},
        Layout<Shape<Int<KT::SFConfig::kNumScaleCopyThreads>>>{}, Layout<Shape<_1>>{});
    auto thr_scale_copy_a = scale_copy_a.get_slice(lane_idx);
    auto thr_scale_copy_b = scale_copy_b.get_slice(lane_idx);
    auto tAgSFA_up = thr_scale_copy_a.partition_S(gSFA_up);
    auto tAgSFA_gate = thr_scale_copy_a.partition_S(gSFA_gate);
    auto tAcSFA_up = thr_scale_copy_a.partition_S(coordSFA_up);
    auto tAcSFA_gate = thr_scale_copy_a.partition_S(coordSFA_gate);
    auto tAsSFA_up = thr_scale_copy_a.partition_D(sSFA_up);
    auto tAsSFA_gate = thr_scale_copy_a.partition_D(sSFA_gate);
    auto tBgSFB = thr_scale_copy_b.partition_S(gSFB);
    auto tBcSFB = thr_scale_copy_b.partition_S(coordSFB);
    auto tBsSFB = thr_scale_copy_b.partition_D(sSFB);

    int64_t scales_m = sm120_common::math::ceil_div(2 * params.N, int(KT::kGranM));
    int64_t scales_n = sf_m_offset / KT::kGranN +
                       sm120_common::math::ceil_div(m_boundary - m_offset, int(KT::kGranN));
    auto tApSFA_up = cute::lazy::transform(tAcSFA_up(_, _, 0), [&](auto const& coord) {
      return lane_idx < KT::SFConfig::kTileScaleM && get<0>(coord) < scales_m;
    });
    auto tApSFA_gate = cute::lazy::transform(tAcSFA_gate(_, _, 0), [&](auto const& coord) {
      return lane_idx < KT::SFConfig::kTileScaleM && get<0>(coord) < scales_m;
    });
    auto tBpSFB = cute::lazy::transform(tBcSFB(_, _, 0), [&](auto const& coord) {
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
      copy_if(scale_copy_a, tApSFA_up, tAgSFA_up(_, _, k_tile_idx), tAsSFA_up(_, _, sf_stage));
      copy_if(scale_copy_a, tApSFA_gate, tAgSFA_gate(_, _, k_tile_idx),
              tAsSFA_gate(_, _, sf_stage));
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

  CUTE_DEVICE
  static void mma_swap(Params const& params, SharedStorage& shared_storage, int32_t k_tile_count,
                       int32_t m_offset, int32_t m_boundary, int32_t m_block_idx,
                       int32_t n_block_idx, int& read_stage, uint32_t& sf_phase,
                       uint32_t& ab_phase) {
    int thread_idx = int(threadIdx.x);

    auto sA_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A_up.begin()),
                    typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sA_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A_gate.begin()),
                    typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sB = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));

    typename KT::MMAConfig::TiledMma mma;
    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto accum_up = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto accum_gate = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto tmp_accum = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto tCrA_up = thr_mma.partition_fragment_A(sA_up(_, _, Int<0>{}));
    auto tCrA_gate = thr_mma.partition_fragment_A(sA_gate(_, _, Int<0>{}));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, Int<0>{}));
    constexpr int K_BLOCK_MAX = decltype(size<2>(tCrA_up))::value;

    auto s2r_copy_A = make_tiled_copy_A(typename KT::ABLoadConfig::SmemCopyAtomA{}, mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
    auto tXsA_up = s2r_thr_copy_A.partition_S(sA_up);
    auto tXsA_gate = s2r_thr_copy_A.partition_S(sA_gate);
    auto tXrA_up = s2r_thr_copy_A.retile_D(tCrA_up);
    auto tXrA_gate = s2r_thr_copy_A.retile_D(tCrA_gate);
    auto s2r_copy_B = make_tiled_copy_B(typename KT::ABLoadConfig::SmemCopyAtomB{}, mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
    auto tXsB = s2r_thr_copy_B.partition_S(sB);
    auto tXrB = s2r_thr_copy_B.retile_D(tCrB);

    auto sSFAViewAsC_up =
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA_up.begin()),
                    typename KT::SFConfig::SmemLayoutSFAViewAsC{});
    auto sSFAViewAsC_gate =
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA_gate.begin()),
                    typename KT::SFConfig::SmemLayoutSFAViewAsC{});
    auto sSFBViewAsC = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                                   typename KT::SFConfig::SmemLayoutSFBViewAsC{});
    auto tCsSFAViewAsC_up = thr_mma.partition_C(sSFAViewAsC_up);
    auto tCsSFAViewAsC_gate = thr_mma.partition_C(sSFAViewAsC_gate);
    auto tCsSFBViewAsC = thr_mma.partition_C(sSFBViewAsC);
    auto tCrSFAViewAsC_up =
        make_tensor_like<typename KT::ElementScale>(tCsSFAViewAsC_up(_, _, _, Int<0>{}));
    auto tCrSFAViewAsC_gate =
        make_tensor_like<typename KT::ElementScale>(tCsSFAViewAsC_gate(_, _, _, Int<0>{}));
    auto tCrSFBViewAsC_up =
        make_tensor_like<typename KT::ElementScale>(tCsSFBViewAsC(_, _, _, Int<0>{}));
    auto tCrSFBViewAsC_gate =
        make_tensor_like<typename KT::ElementScale>(tCsSFBViewAsC(_, _, _, Int<0>{}));

    clear(accum_up);
    clear(accum_gate);
    clear(tmp_accum);
    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = get_mbarriers(shared_storage);
    (void)store_full_mbar;

    auto copy_ab_s2r = [&](auto k_block) {
      cute::copy(s2r_copy_A, tXsA_up(_, _, k_block, read_stage), tXrA_up(_, _, k_block));
      cute::copy(s2r_copy_A, tXsA_gate(_, _, k_block, read_stage), tXrA_gate(_, _, k_block));
      cute::copy(s2r_copy_B, tXsB(_, _, k_block, read_stage), tXrB(_, _, k_block));
    };
    auto advance_read_stage = [&]() {
      read_stage = (read_stage + 1) % KT::AB_Stages;
      if (read_stage == 0) {
        ab_phase ^= 1;
        sf_phase ^= 1;
      }
    };
    auto copy_scale_s2r = [&]() {
      sf_full_mbar[read_stage].wait(sf_phase);
      cute::copy(tCsSFAViewAsC_up(_, _, _, read_stage), tCrSFAViewAsC_up);
      cute::copy(tCsSFAViewAsC_gate(_, _, _, read_stage), tCrSFAViewAsC_gate);
      cute::copy(tCsSFBViewAsC(_, _, _, read_stage), tCrSFBViewAsC_up);
      cute::copy(tCsSFBViewAsC(_, _, _, read_stage), tCrSFBViewAsC_gate);
      sf_empty_mbar[read_stage].arrive();
      typename KT::ElementScale scale_a_up = tCrSFAViewAsC_up.data()[0];
      typename KT::ElementScale scale_a_gate = tCrSFAViewAsC_gate.data()[0];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCrSFBViewAsC_up); ++i) {
        tCrSFBViewAsC_up.data()[i] *= scale_a_up;
        tCrSFBViewAsC_gate.data()[i] *= scale_a_gate;
      }
    };
    auto rescale_up = [&]() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum_up); ++i) {
        accum_up(i) += tmp_accum(i) * tCrSFBViewAsC_up(i);
        tmp_accum(i) = 0.0f;
      }
    };
    auto rescale_gate = [&]() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum_gate); ++i) {
        accum_gate(i) += tmp_accum(i) * tCrSFBViewAsC_gate(i);
        tmp_accum(i) = 0.0f;
      }
    };

    copy_scale_s2r();
    ab_full_mbar[read_stage].wait(ab_phase);
    copy_ab_s2r(Int<0>{});

    for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count - 1; ++k_tile_idx) {
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        constexpr int k_block_idx = decltype(k_block)::value;
        if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
          copy_ab_s2r(Int<k_block_idx + 1>{});
        }
        cute::gemm(mma, tCrA_up(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
      });
      rescale_up();

      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        cute::gemm(mma, tCrA_gate(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
      });
      ab_empty_mbar[read_stage].arrive();
      advance_read_stage();
      ab_full_mbar[read_stage].wait(ab_phase);
      copy_ab_s2r(Int<0>{});
      rescale_gate();
      copy_scale_s2r();
    }

    cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      constexpr int k_block_idx = decltype(k_block)::value;
      if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
        copy_ab_s2r(Int<k_block_idx + 1>{});
      }
      cute::gemm(mma, tCrA_up(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
    });
    rescale_up();

    cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      cute::gemm(mma, tCrA_gate(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
    });
    ab_empty_mbar[read_stage].arrive();
    advance_read_stage();
    rescale_gate();

    cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
    CUTE_UNROLL
    for (int i = 0; i < size(accum_gate); ++i) {
      tmp_accum(i) = __expf(-accum_gate(i));
    }
    CUTE_UNROLL
    for (int i = 0; i < size(accum_gate); ++i) {
      accum_gate(i) = accum_gate(i) / (1.0f + tmp_accum(i));
    }
    CUTE_UNROLL
    for (int i = 0; i < size(accum_up); ++i) {
      accum_up(i) *= accum_gate(i);
    }
    sm120_common::utils::epi_pred_stg<KT>(params, accum_up, thread_idx, m_offset, m_boundary,
                                          m_block_idx, n_block_idx, store_empty_mbar);
  }

  CUTE_DEVICE
  static void mma(Params const& params, SharedStorage& shared_storage, int32_t k_tile_count,
                  int32_t m_offset, int32_t m_boundary, int32_t m_block_idx, int32_t n_block_idx,
                  int32_t expert_idx, int& read_stage, uint32_t& sf_phase, uint32_t& ab_phase,
                  int& epi_stage, uint32_t& se_phase) {
    int thread_idx = int(threadIdx.x);

    auto sA = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
                    typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sB_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_up.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto sB_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_gate.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));

    typename KT::MMAConfig::TiledMma mma;
    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto accum_up = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto accum_gate = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto tmp_accum = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    clear(accum_up);
    clear(accum_gate);
    clear(tmp_accum);

    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    auto tCrB = thr_mma.partition_fragment_B(sB_up(_, _, Int<0>{}));
    constexpr int K_BLOCK_MAX = decltype(size<2>(tCrA))::value;
    auto s2r_copy_A = make_tiled_copy_A(typename KT::ABLoadConfig::SmemCopyAtomA{}, mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
    auto tXsA = s2r_thr_copy_A.partition_S(sA);
    auto tXrA = s2r_thr_copy_A.retile_D(tCrA);
    auto s2r_copy_B = make_tiled_copy_B(typename KT::ABLoadConfig::SmemCopyAtomB{}, mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
    auto tXsB_up = s2r_thr_copy_B.partition_S(sB_up);
    auto tXsB_gate = s2r_thr_copy_B.partition_S(sB_gate);
    auto tXrB = s2r_thr_copy_B.retile_D(tCrB);

    auto sSFAViewAsC = make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                                   typename KT::SFConfig::SmemLayoutSFAViewAsC{});
    auto sSFBViewAsC_up =
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_up.begin()),
                    typename KT::SFConfig::SmemLayoutSFBViewAsC{});
    auto sSFBViewAsC_gate =
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_gate.begin()),
                    typename KT::SFConfig::SmemLayoutSFBViewAsC{});
    auto tCsSFAViewAsC = thr_mma.partition_C(sSFAViewAsC);
    auto tCsSFBViewAsC_up = thr_mma.partition_C(sSFBViewAsC_up);
    auto tCsSFBViewAsC_gate = thr_mma.partition_C(sSFBViewAsC_gate);
    auto tCrSFAViewAsC_up =
        make_tensor_like<typename KT::ElementScale>(tCsSFAViewAsC(_, _, _, Int<0>{}));
    auto tCrSFAViewAsC_gate =
        make_tensor_like<typename KT::ElementScale>(tCsSFAViewAsC(_, _, _, Int<0>{}));
    auto tCrSFBViewAsC_up =
        make_tensor_like<typename KT::ElementScale>(tCsSFBViewAsC_up(_, _, _, Int<0>{}));
    auto tCrSFBViewAsC_gate =
        make_tensor_like<typename KT::ElementScale>(tCsSFBViewAsC_gate(_, _, _, Int<0>{}));

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = get_mbarriers(shared_storage);

    auto rescale_up = [&]() {
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN == 1) {
        typename KT::ElementScale scale_ab = tCrSFAViewAsC_up.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_up); ++i) {
          accum_up(i) += tmp_accum(i) * scale_ab;
          tmp_accum(i) = 0.0f;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN == 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_up); ++i) {
          accum_up(i) += tmp_accum(i) * tCrSFAViewAsC_up(i);
          tmp_accum(i) = 0.0f;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_up); ++i) {
          accum_up(i) += tmp_accum(i) * tCrSFBViewAsC_up(i);
          tmp_accum(i) = 0.0f;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_up); ++i) {
          accum_up(i) += tmp_accum(i) * tCrSFAViewAsC_up(i) * tCrSFBViewAsC_up(i);
          tmp_accum(i) = 0.0f;
        }
      }
    };
    auto rescale_gate = [&]() {
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN == 1) {
        typename KT::ElementScale scale_ab = tCrSFAViewAsC_gate.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_gate); ++i) {
          accum_gate(i) += tmp_accum(i) * scale_ab;
          tmp_accum(i) = 0.0f;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN == 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_gate); ++i) {
          accum_gate(i) += tmp_accum(i) * tCrSFAViewAsC_gate(i);
          tmp_accum(i) = 0.0f;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_gate); ++i) {
          accum_gate(i) += tmp_accum(i) * tCrSFBViewAsC_gate(i);
          tmp_accum(i) = 0.0f;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN > 1) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum_gate); ++i) {
          accum_gate(i) += tmp_accum(i) * tCrSFAViewAsC_gate(i) * tCrSFBViewAsC_gate(i);
          tmp_accum(i) = 0.0f;
        }
      }
    };
    auto tXsA_stage = tXsA(_, _, _, read_stage);
    auto tXsB_up_stage = tXsB_up(_, _, _, read_stage);
    auto tXsB_gate_stage = tXsB_gate(_, _, _, read_stage);
    auto copy_scale_s2r = [&]() {
      cute::copy(tCsSFAViewAsC(_, _, _, read_stage), tCrSFAViewAsC_up);
      cute::copy(tCsSFAViewAsC(_, _, _, read_stage), tCrSFAViewAsC_gate);
      cute::copy(tCsSFBViewAsC_up(_, _, _, read_stage), tCrSFBViewAsC_up);
      cute::copy(tCsSFBViewAsC_gate(_, _, _, read_stage), tCrSFBViewAsC_gate);
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN == 1) {
        tCrSFAViewAsC_up.data()[0] *= tCrSFBViewAsC_up.data()[0];
        tCrSFAViewAsC_gate.data()[0] *= tCrSFBViewAsC_gate.data()[0];
      }
      if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN == 1) {
        typename KT::ElementScale scale_b_up = tCrSFBViewAsC_up.data()[0];
        typename KT::ElementScale scale_b_gate = tCrSFBViewAsC_gate.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrSFAViewAsC_up); ++i) {
          tCrSFAViewAsC_up.data()[i] *= scale_b_up;
          tCrSFAViewAsC_gate.data()[i] *= scale_b_gate;
        }
      }
      if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN > 1) {
        typename KT::ElementScale scale_a_up = tCrSFAViewAsC_up.data()[0];
        typename KT::ElementScale scale_a_gate = tCrSFAViewAsC_gate.data()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tCrSFBViewAsC_up); ++i) {
          tCrSFBViewAsC_up.data()[i] *= scale_a_up;
          tCrSFBViewAsC_gate.data()[i] *= scale_a_gate;
        }
      }
    };
    auto copy_up_s2r = [&](auto k_block) {
      cute::copy(s2r_copy_A, tXsA_stage(_, _, k_block), tXrA(_, _, k_block));
      cute::copy(s2r_copy_B, tXsB_up_stage(_, _, k_block), tXrB(_, _, k_block));
    };
    auto copy_gate_s2r = [&](auto k_block) {
      cute::copy(s2r_copy_B, tXsB_gate_stage(_, _, k_block), tXrB(_, _, k_block));
    };
    auto advance_read_stage = [&]() {
      ++read_stage;
      if (read_stage == KT::AB_Stages) {
        read_stage = 0;
        ab_phase ^= 1;
        sf_phase ^= 1;
      }
      tXsA_stage = tXsA(_, _, _, read_stage);
      tXsB_up_stage = tXsB_up(_, _, _, read_stage);
      tXsB_gate_stage = tXsB_gate(_, _, _, read_stage);
    };

    sf_full_mbar[read_stage].wait(sf_phase);
    copy_scale_s2r();
    sf_empty_mbar[read_stage].arrive();
    ab_full_mbar[read_stage].wait(ab_phase);
    copy_up_s2r(Int<0>{});

    for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count - 1; ++k_tile_idx) {
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        constexpr int k_block_idx = decltype(k_block)::value;
        if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
          copy_up_s2r(Int<k_block_idx + 1>{});
        }
        cute::gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
      });
      rescale_up();

      copy_gate_s2r(Int<0>{});
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        constexpr int k_block_idx = decltype(k_block)::value;
        if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
          copy_gate_s2r(Int<k_block_idx + 1>{});
        } else {
          ab_empty_mbar[read_stage].arrive();
          advance_read_stage();
          ab_full_mbar[read_stage].wait(ab_phase);
          copy_up_s2r(Int<0>{});
        }
        cute::gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
      });
      rescale_gate();

      sf_full_mbar[read_stage].wait(sf_phase);
      copy_scale_s2r();
      sf_empty_mbar[read_stage].arrive();
    }

    cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      constexpr int k_block_idx = decltype(k_block)::value;
      if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
        copy_up_s2r(Int<k_block_idx + 1>{});
      }
      cute::gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
    });
    rescale_up();

    copy_gate_s2r(Int<0>{});
    cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
      constexpr int k_block_idx = decltype(k_block)::value;
      if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
        copy_gate_s2r(Int<k_block_idx + 1>{});
      } else {
        ab_empty_mbar[read_stage].arrive();
        advance_read_stage();
      }
      cute::gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tmp_accum);
    });
    if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN == 1) {
      typename KT::ElementScale scale_ab = tCrSFAViewAsC_gate.data()[0];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum_gate); ++i) {
        accum_gate(i) += tmp_accum(i) * scale_ab;
        tmp_accum(i) = __expf(-accum_gate(i));
      }
    }
    if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN == 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum_gate); ++i) {
        accum_gate(i) += tmp_accum(i) * tCrSFAViewAsC_gate(i);
        tmp_accum(i) = __expf(-accum_gate(i));
      }
    }
    if constexpr (KT::SFConfig::kTileScaleM == 1 && KT::SFConfig::kTileScaleN > 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum_gate); ++i) {
        accum_gate(i) += tmp_accum(i) * tCrSFBViewAsC_gate(i);
        tmp_accum(i) = __expf(-accum_gate(i));
      }
    }
    if constexpr (KT::SFConfig::kTileScaleM > 1 && KT::SFConfig::kTileScaleN > 1) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum_gate); ++i) {
        accum_gate(i) += tmp_accum(i) * tCrSFAViewAsC_gate(i) * tCrSFBViewAsC_gate(i);
        tmp_accum(i) = __expf(-accum_gate(i));
      }
    }

    CUTE_UNROLL
    for (int i = 0; i < size(accum_gate); ++i) {
      accum_gate(i) = accum_gate(i) / (1.0f + tmp_accum(i));
    }

    CUTE_UNROLL
    for (int i = 0; i < size(accum_up); ++i) {
      accum_up(i) *= accum_gate(i);
    }
    if constexpr (kUseStagedR2G) {
      sm120_common::utils::epi_staged_r2s<KT>(params, shared_storage, accum_up, thread_idx,
                                              m_offset, m_boundary, m_block_idx, epi_stage,
                                              se_phase, store_full_mbar, store_empty_mbar);
    } else {
      sm120_common::utils::epi_pred_r2g<KT>(params, shared_storage, accum_up, thread_idx, m_offset,
                                            m_boundary, m_block_idx, n_block_idx, expert_idx,
                                            store_empty_mbar);
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
      cute::for_each(cute::make_int_sequence<KT::AB_Stages>{}, [&](auto stage) {
        ab_full_mbar[stage].init(1);
        ab_empty_mbar[stage].init(KT::MMAConfig::kNumMathThreads);
      });
      cute::for_each(cute::make_int_sequence<KT::SFConfig::SF_Stages>{}, [&](auto stage) {
        sf_full_mbar[stage].init(KT::kNumProducerThreadEvents);
        sf_empty_mbar[stage].init(KT::MMAConfig::kNumMathThreads);
      });
      if constexpr (kUseStagedR2G) {
        cute::for_each(cute::make_int_sequence<KT::StagedR2GStoreConfig::StagesD>{},
                       [&](auto stage) {
                         store_full_mbar[stage].init(KT::MMAConfig::kNumMathThreads);
                         store_empty_mbar[stage].init(KT::StagedR2GStoreConfig::kNumStoreThreads);
                       });
      } else {
        store_empty_mbar[0].init(KT::MMAConfig::kNumMathThreads);
      }
      shared_storage.sched.init_mbars(kNumSchedConsumers);
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    int32_t k_tile_count = sm120_common::math::ceil_div(params.K, int(KT::kTileK));

    if (warp_idx >= KT::MMAConfig::kNumMathWarps) {
      cutlass::arch::warpgroup_reg_dealloc<KT::LoadRegisterRequirement>();
      constexpr int sched_warp_idx = KT::MMAConfig::kNumMathWarps;
      constexpr int ab_warp_idx = sched_warp_idx + 1;
      constexpr int sf_warp_idx = ab_warp_idx + 1;
      constexpr int store_warp_idx = sf_warp_idx + 1;

      if (warp_idx == ab_warp_idx) {
        int ab_stage = 0;
        uint32_t ab_phase = 1;
        uint32_t store_phase = 1;
        sm120_common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                       lane_predicate};
        sm120_common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          if (lane_predicate) {
            auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
                tile.m_block, tile.n_block, tile.group);
            if constexpr (KT::kSwapAB) {
              load_ab_swap(params, shared_storage, blk_coord, tile.m_offset, k_tile_count, ab_stage,
                           ab_phase, store_phase);
            } else {
              load_ab(params, shared_storage, blk_coord, tile.m_offset, k_tile_count, ab_stage,
                      ab_phase, store_phase);
            }
          }
        }
      } else if (warp_idx == sf_warp_idx) {
        int sf_stage = 0;
        uint32_t sf_phase = 1;
        uint32_t store_phase = 1;
        sm120_common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                       lane_predicate};
        sm120_common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
              tile.m_block, tile.n_block, tile.group);
          if constexpr (KT::kSwapAB) {
            load_sf_swap(params, shared_storage, blk_coord, tile.m_offset, tile.m_boundary,
                         k_tile_count, sf_stage, sf_phase, store_phase);
          } else {
            load_sf(params, shared_storage, blk_coord, tile.m_offset, k_tile_count, sf_stage,
                    sf_phase, store_phase);
          }
        }
      } else if (warp_idx == sched_warp_idx) {
        Scheduler scheduler(params.M, params.N, params.num_experts, params.grouped_layout);
        sm120_common::MoeSchedProducer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                       lane_predicate};
        int32_t m_block_idx;
        int32_t n_block_idx;
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
          sched_pipeline.publish(sm120_common::MoeWorkTile{
              m_block_idx, n_block_idx, scheduler.get_expert_idx(m_block_idx),
              scheduler.get_m_offset(), scheduler.get_m_boundary(), 1});
        }
        sched_pipeline.publish_sentinel();
      } else if constexpr (kUseStagedR2G) {
        if (warp_idx == store_warp_idx) {
          uint32_t full_phase = 0;
          int store_stage = 0;
          int store_thread_idx = cutlass::canonical_lane_idx();
          sm120_common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                         lane_predicate};
          sm120_common::MoeWorkTile tile;
          while (sched_pipeline.get_next_tile(tile)) {
            sm120_common::utils::staged_r2g_store<KT>(
                params, shared_storage, tile.m_offset, tile.m_boundary, tile.m_block, tile.n_block,
                store_thread_idx, full_phase, store_stage, store_full_mbar, store_empty_mbar);
          }
        }
      }
    } else {
      cutlass::arch::warpgroup_reg_alloc<KT::MmaRegisterRequirement>();
      int read_stage = 0;
      uint32_t sf_phase = 0;
      uint32_t ab_phase = 0;
      int epi_stage = 0;
      uint32_t se_phase = 1;
      sm120_common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                     lane_predicate};
      sm120_common::MoeWorkTile tile;
      while (sched_pipeline.get_next_tile(tile)) {
        auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(tile.m_block,
                                                                          tile.n_block, tile.group);
        if constexpr (KT::kSwapAB) {
          mma_swap(params, shared_storage, k_tile_count, tile.m_offset, tile.m_boundary,
                   get<0>(blk_coord), get<1>(blk_coord), read_stage, sf_phase, ab_phase);
        } else {
          mma(params, shared_storage, k_tile_count, tile.m_offset, tile.m_boundary,
              get<0>(blk_coord), get<1>(blk_coord), get<2>(blk_coord), read_stage, sf_phase,
              ab_phase, epi_stage, se_phase);
        }
      }
    }
  }
};

}  // namespace sm120_blockscaling
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
