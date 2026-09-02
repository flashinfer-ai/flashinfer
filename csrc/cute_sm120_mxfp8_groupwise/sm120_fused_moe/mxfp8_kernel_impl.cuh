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
#include <cutlass/epilogue/thread/activation.h>

#include "cute_sm120_mxfp8_groupwise/sm120_fused_moe/mxfp8_builder.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_blockscaled/kernel_impl.cuh"
#include "cute_sm120_mxfp8_groupwise/sm120_common/moe_scheduler.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaled {

using namespace cute;

template <typename KT>
struct SM120BlockScaledFusedMoeGemmKernel : SM120BlockScaledGemmKernel<KT> {
  using BaseKernel = SM120BlockScaledGemmKernel<KT>;
  using Scheduler = sm120_common::SelectedMoeScheduler<KT::kSwapAB, KT::kTileM, KT::kTileN>;
  using ProblemShape = typename BaseKernel::ProblemShape;
  static constexpr int kNumSchedStages = 2;
  static constexpr int kNumSchedConsumers =
      KT::MMAConfig::kNumMathWarps + (KT::kUseStagedR2G ? 3 : 2);
  struct SharedStorage : BaseKernel::SharedStorage {
    alignas(8) sm120_common::MoeSchedStorage<kNumSchedStages> sched;
  };
  static constexpr int kSmemSize = int(sizeof(SharedStorage));
  using FullBarrier = typename KT::FullBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  static constexpr bool kUseStagedR2G = KT::kUseStagedR2G;
  static constexpr uint32_t TmaTransactionBytesAB =
      KT::kSwapAB
          ? 2 * KT::ABLoadConfig::TmaTransactionBytesA + KT::ABLoadConfig::TmaTransactionBytesB
          : KT::ABLoadConfig::TmaTransactionBytesA + 2 * KT::ABLoadConfig::TmaTransactionBytesB;
  static constexpr uint32_t TmaTransactionBytesSF =
      KT::kSwapAB ? 2 * KT::SFConfig::TmaTransactionBytesSFA + KT::SFConfig::TmaTransactionBytesSFB
                  : KT::SFConfig::TmaTransactionBytesSFA + 2 * KT::SFConfig::TmaTransactionBytesSFB;

  struct Params {
    typename KT::ABLoadConfig::TMA_A tma_load_a;
    typename KT::ABLoadConfig::TMA_B tma_load_b;
    typename KT::SFConfig::TMA_SFA tma_load_sfa;
    typename KT::SFConfig::TMA_SFB tma_load_sfb;
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
  };

  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args) {
    auto [M, N, K, num_experts] = problem_shape;
    auto [tma_load_a, tma_load_b] = sm120_common::utils::make_ab_tma_descriptors<KT>(
        args.ptr_A, args.dA, args.ptr_B, args.dB, M, 2 * N, K, num_experts);
    auto [tma_load_sfa, tma_load_sfb] =
        utils::make_sf_tma_descriptors<KT>(args.ptr_SFA, args.ptr_SFB, M, 2 * N, K, num_experts);
    return {tma_load_a, tma_load_b, tma_load_sfa, tma_load_sfb,       args.ptr_D, M,
            N,          K,          num_experts,  args.grouped_layout};
  }

  CUTE_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfa.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_sfb.get_tma_descriptor());
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_ab(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset,
                                  int32_t num_sf_cycles, uint32_t& ab_phase,
                                  uint32_t& store_phase) {
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
          store_empty_mbar] = BaseKernel::get_mbarriers(shared_storage);
    (void)sf_full_mbar;
    (void)sf_empty_mbar;
    (void)store_full_mbar;

    int32_t k_tile_count =
        num_sf_cycles * KT::SFConfig::SF_Stages * KT::SFConfig::kNumTileKPerPackSF;

    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    for (int32_t k_tile_idx = 0; k_tile_idx < k_tile_count; k_tile_idx += KT::AB_Stages) {
      cute::for_each(cute::make_int_sequence<KT::AB_Stages>{}, [&](auto write_stage) {
        ab_empty_mbar[write_stage].wait(ab_phase);
        auto& ab_full_barrier = ab_full_mbar[write_stage];
        ab_full_barrier.arrive_and_expect_tx(TmaTransactionBytesAB);
        auto tma_copy_a =
            params.tma_load_a.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
        cute::copy(tma_copy_a, tAgA(_, _, _, k_tile_idx + write_stage), tAsA(_, _, _, write_stage));
        auto tma_copy_b =
            params.tma_load_b.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
        cute::copy(tma_copy_b, tBgB_up(_, _, _, k_tile_idx + write_stage),
                   tBsB_up(_, _, _, write_stage));
        cute::copy(tma_copy_b, tBgB_gate(_, _, _, k_tile_idx + write_stage),
                   tBsB_gate(_, _, _, write_stage));
      });
      ab_phase ^= 1;
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_ab_swap(Params const& params, SharedStorage& shared_storage,
                                       BlkCoord const& blk_coord, int32_t m_offset,
                                       int32_t num_sf_cycles, uint32_t& ab_phase,
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
          store_empty_mbar] = BaseKernel::get_mbarriers(shared_storage);
    (void)sf_full_mbar;
    (void)sf_empty_mbar;
    (void)store_full_mbar;

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
        ab_full_barrier.arrive_and_expect_tx(TmaTransactionBytesAB);
        auto tma_copy_a =
            params.tma_load_a.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
        cute::copy(tma_copy_a, tAgA_up(_, _, _, k_tile_idx + write_stage),
                   tAsA_up(_, _, _, write_stage));
        cute::copy(tma_copy_a, tAgA_gate(_, _, _, k_tile_idx + write_stage),
                   tAsA_gate(_, _, _, write_stage));
        auto tma_copy_b =
            params.tma_load_b.with(*recast_ptr<ProducerBarrierType>(&ab_full_barrier));
        cute::copy(tma_copy_b, tBgB(_, _, _, k_tile_idx + write_stage), tBsB(_, _, _, write_stage));
      });
      ab_phase ^= 1;
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_sf(Params const& params, SharedStorage& shared_storage,
                                  BlkCoord const& blk_coord, int32_t m_offset,
                                  int32_t num_sf_cycles, uint32_t& sf_phase,
                                  uint32_t& store_phase) {
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    using X = Underscore;

    auto sf_shape = make_shape(params.M, 2 * params.N, params.K, 1);
    auto sfa_layout = KT::SFConfig::deduce_sfa_layout(sf_shape, params.num_experts);
    auto mSFA_full = params.tma_load_sfa.get_tma_tensor(shape(sfa_layout));
    int32_t sf_m_offset = sm120_common::math::compute_padded_offset(m_offset, expert_idx);
    auto mSFA = domain_offset(make_coord(sf_m_offset, 0, 0), mSFA_full);
    auto gSFA_mkl = local_tile(mSFA, typename KT::SFConfig::ScaleTileShape{}, make_coord(_, _, _),
                               Step<_1, X, _1>{});
    auto gSFA = gSFA_mkl(_, _, m_block_idx, _, Int<0>{});

    auto sfb_shape = make_shape(params.M, 2 * params.N, params.K, params.num_experts);
    auto sfb_layout = KT::SFConfig::deduce_sfb_layout(sfb_shape, params.num_experts);
    auto mSFB = params.tma_load_sfb.get_tma_tensor(shape(sfb_layout));
    auto mSFB_up = mSFB;
    auto mSFB_gate = domain_offset(make_coord(params.N, 0, 0), mSFB);
    auto gSFB_up_nkl = local_tile(mSFB_up, typename KT::SFConfig::ScaleTileShape{},
                                  make_coord(_, _, _), Step<X, _1, _1>{});
    auto gSFB_gate_nkl = local_tile(mSFB_gate, typename KT::SFConfig::ScaleTileShape{},
                                    make_coord(_, _, _), Step<X, _1, _1>{});
    auto gSFB_up = gSFB_up_nkl(_, _, n_block_idx, _, expert_idx);
    auto gSFB_gate = gSFB_gate_nkl(_, _, n_block_idx, _, expert_idx);

    auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
    auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
    auto tAgSFA = block_tma_sfa.partition_S(gSFA);
    auto tBgSFB_up = block_tma_sfb.partition_S(gSFB_up);
    auto tBgSFB_gate = block_tma_sfb.partition_S(gSFB_gate);

    auto sSFA = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                    typename KT::SFConfig::SmemLayoutSFA{}));
    auto sSFB_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_up.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));
    auto sSFB_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_gate.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));
    auto tAsSFA = block_tma_sfa.partition_D(sSFA);
    auto tBsSFB_up = block_tma_sfb.partition_D(sSFB_up);
    auto tBsSFB_gate = block_tma_sfb.partition_D(sSFB_gate);

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = BaseKernel::get_mbarriers(shared_storage);
    (void)ab_full_mbar;
    (void)ab_empty_mbar;
    (void)store_full_mbar;

    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    for (int32_t sf_cycle = 0; sf_cycle < num_sf_cycles; ++sf_cycle) {
      cute::for_each(cute::make_int_sequence<KT::SFConfig::SF_Stages>{}, [&](auto sf_stage) {
        sf_empty_mbar[sf_stage].wait(sf_phase);
        auto& sf_full_barrier = sf_full_mbar[sf_stage];
        sf_full_barrier.arrive_and_expect_tx(TmaTransactionBytesSF);
        auto tma_copy_sfa =
            params.tma_load_sfa.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
        auto tma_copy_sfb =
            params.tma_load_sfb.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
        int32_t sf_tile_idx = sf_cycle * KT::SFConfig::SF_Stages + sf_stage;
        cute::copy(tma_copy_sfa, tAgSFA(_, _, _, sf_tile_idx), tAsSFA(_, _, _, sf_stage));
        cute::copy(tma_copy_sfb, tBgSFB_up(_, _, _, sf_tile_idx), tBsSFB_up(_, _, _, sf_stage));
        cute::copy(tma_copy_sfb, tBgSFB_gate(_, _, _, sf_tile_idx), tBsSFB_gate(_, _, _, sf_stage));
      });
      sf_phase ^= 1;
    }
  }

  template <typename BlkCoord>
  CUTE_DEVICE static void load_sf_swap(Params const& params, SharedStorage& shared_storage,
                                       BlkCoord const& blk_coord, int32_t m_offset,
                                       int32_t num_sf_cycles, uint32_t& sf_phase,
                                       uint32_t& store_phase) {
    auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
    using X = Underscore;

    auto sf_shape = make_shape(2 * params.N, params.M, params.K, params.num_experts);
    auto sfa_layout = KT::SFConfig::deduce_sfa_layout(sf_shape, params.num_experts);
    auto mSFA = params.tma_load_sfa.get_tma_tensor(shape(sfa_layout));
    auto mSFA_up = mSFA;
    auto mSFA_gate = domain_offset(make_coord(params.N, 0, 0), mSFA);
    auto gSFA_up_mkl = local_tile(mSFA_up, typename KT::SFConfig::ScaleTileShape{},
                                  make_coord(_, _, _), Step<_1, X, _1>{});
    auto gSFA_gate_mkl = local_tile(mSFA_gate, typename KT::SFConfig::ScaleTileShape{},
                                    make_coord(_, _, _), Step<_1, X, _1>{});
    auto gSFA_up = gSFA_up_mkl(_, _, m_block_idx, _, expert_idx);
    auto gSFA_gate = gSFA_gate_mkl(_, _, m_block_idx, _, expert_idx);

    auto sfb_layout = KT::SFConfig::deduce_sfb_layout(
        make_shape(2 * params.N, params.M, params.K, 1), params.num_experts);
    auto mSFB_full = params.tma_load_sfb.get_tma_tensor(shape(sfb_layout));
    int32_t sf_m_offset = sm120_common::math::compute_padded_offset(m_offset, expert_idx);
    auto mSFB = domain_offset(make_coord(sf_m_offset, 0, 0), mSFB_full);
    auto gSFB_nkl = local_tile(mSFB, typename KT::SFConfig::ScaleTileShape{}, make_coord(_, _, _),
                               Step<X, _1, _1>{});
    auto gSFB = gSFB_nkl(_, _, n_block_idx, _, Int<0>{});

    auto block_tma_sfa = params.tma_load_sfa.get_slice(0);
    auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
    auto tAgSFA_up = block_tma_sfa.partition_S(gSFA_up);
    auto tAgSFA_gate = block_tma_sfa.partition_S(gSFA_gate);
    auto tBgSFB = block_tma_sfb.partition_S(gSFB);

    auto sSFA_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA_up.begin()),
                    typename KT::SFConfig::SmemLayoutSFA{}));
    auto sSFA_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA_gate.begin()),
                    typename KT::SFConfig::SmemLayoutSFA{}));
    auto sSFB = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));
    auto tAsSFA_up = block_tma_sfa.partition_D(sSFA_up);
    auto tAsSFA_gate = block_tma_sfa.partition_D(sSFA_gate);
    auto tBsSFB = block_tma_sfb.partition_D(sSFB);

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = BaseKernel::get_mbarriers(shared_storage);
    (void)ab_full_mbar;
    (void)ab_empty_mbar;
    (void)store_full_mbar;

    if constexpr (KT::kUnionSmem) {
      store_empty_mbar[0].wait(store_phase);
      store_phase ^= 1;
    }

    for (int32_t sf_cycle = 0; sf_cycle < num_sf_cycles; ++sf_cycle) {
      cute::for_each(cute::make_int_sequence<KT::SFConfig::SF_Stages>{}, [&](auto sf_stage) {
        sf_empty_mbar[sf_stage].wait(sf_phase);
        auto& sf_full_barrier = sf_full_mbar[sf_stage];
        sf_full_barrier.arrive_and_expect_tx(TmaTransactionBytesSF);
        auto tma_copy_sfa =
            params.tma_load_sfa.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
        auto tma_copy_sfb =
            params.tma_load_sfb.with(*recast_ptr<ProducerBarrierType>(&sf_full_barrier));
        int32_t sf_tile_idx = sf_cycle * KT::SFConfig::SF_Stages + sf_stage;
        cute::copy(tma_copy_sfa, tAgSFA_up(_, _, _, sf_tile_idx), tAsSFA_up(_, _, _, sf_stage));
        cute::copy(tma_copy_sfa, tAgSFA_gate(_, _, _, sf_tile_idx), tAsSFA_gate(_, _, _, sf_stage));
        cute::copy(tma_copy_sfb, tBgSFB(_, _, _, sf_tile_idx), tBsSFB(_, _, _, sf_stage));
      });
      sf_phase ^= 1;
    }
  }

  CUTE_DEVICE
  static void mma_swap(Params const& params, SharedStorage& shared_storage, int32_t num_sf_cycles,
                       int32_t m_offset, int32_t m_boundary, int32_t m_block_idx,
                       int32_t n_block_idx, uint32_t& sf_phase, uint32_t& ab_phase) {
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
    auto sSFA_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA_up.begin()),
                    typename KT::SFConfig::SmemLayoutSFA{}));
    auto sSFA_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA_gate.begin()),
                    typename KT::SFConfig::SmemLayoutSFA{}));
    auto sSFB = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));

    typename KT::MMAConfig::TiledMma mma;
    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto accum_up = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto accum_gate = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
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

    auto s2r_copy_SFA = make_tiled_copy_impl(typename KT::SFConfig::SmemCopyAtomSF{},
                                             KT::SFConfig::get_layoutSFA_TV(mma),
                                             make_shape(size<0>(tile_shape(mma)), _1{}));
    auto s2r_thr_copy_SFA = s2r_copy_SFA.get_thread_slice(thread_idx);
    auto tXsSFA_up = s2r_thr_copy_SFA.partition_S(sSFA_up);
    auto tXsSFA_gate = s2r_thr_copy_SFA.partition_S(sSFA_gate);
    auto tCrSFA_up = KT::SFConfig::partition_fragment_SFA(sSFA_up(_, _, Int<0>{}), thr_mma);
    auto tCrSFA_gate = KT::SFConfig::partition_fragment_SFA(sSFA_gate(_, _, Int<0>{}), thr_mma);
    auto tXrSFA_up = s2r_thr_copy_SFA.retile_D(tCrSFA_up);
    auto tXrSFA_gate = s2r_thr_copy_SFA.retile_D(tCrSFA_gate);
    auto tCrSFA_up_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFA_up);
    auto tCrSFA_gate_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFA_gate);

    auto s2r_copy_SFB = make_tiled_copy_impl(typename KT::SFConfig::SmemCopyAtomSF{},
                                             KT::SFConfig::get_layoutSFB_TV(mma),
                                             make_shape(size<1>(tile_shape(mma)), _1{}));
    auto s2r_thr_copy_SFB = s2r_copy_SFB.get_thread_slice(thread_idx);
    auto tXsSFB = s2r_thr_copy_SFB.partition_S(sSFB);
    auto tCrSFB = KT::SFConfig::partition_fragment_SFB(sSFB(_, _, Int<0>{}), thr_mma);
    auto tXrSFB = s2r_thr_copy_SFB.retile_D(tCrSFB);
    auto tCrSFB_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFB);

    clear(accum_up);
    clear(accum_gate);
    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = BaseKernel::get_mbarriers(shared_storage);
    (void)store_full_mbar;

    int ab_read_stage = 0;
    int sf_read_stage = 0;

    auto copy_ab_s2r = [&](auto k_block) {
      cute::copy(s2r_copy_A, tXsA_up(_, _, k_block, ab_read_stage), tXrA_up(_, _, k_block));
      cute::copy(s2r_copy_A, tXsA_gate(_, _, k_block, ab_read_stage), tXrA_gate(_, _, k_block));
      cute::copy(s2r_copy_B, tXsB(_, _, k_block, ab_read_stage), tXrB(_, _, k_block));
    };
    auto advance_ab_read_stage = [&]() {
      ab_read_stage = (ab_read_stage + 1) % KT::AB_Stages;
      ab_phase ^= uint32_t(ab_read_stage == 0);
    };
    auto copy_sf_s2r = [&]() {
      if constexpr (KT::SFConfig::SF_Stages == 1) {
        sf_full_mbar[0].wait(sf_phase);
        cute::copy(s2r_copy_SFA, tXsSFA_up(_, _, _, Int<0>{}), tXrSFA_up);
        cute::copy(s2r_copy_SFA, tXsSFA_gate(_, _, _, Int<0>{}), tXrSFA_gate);
        cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, Int<0>{}), tXrSFB);
        sf_empty_mbar[0].arrive();
        sf_phase ^= 1;
      } else {
        sf_full_mbar[sf_read_stage].wait(sf_phase);
        cute::copy(s2r_copy_SFA, tXsSFA_up(_, _, _, sf_read_stage), tXrSFA_up);
        cute::copy(s2r_copy_SFA, tXsSFA_gate(_, _, _, sf_read_stage), tXrSFA_gate);
        cute::copy(s2r_copy_SFB, tXsSFB(_, _, _, sf_read_stage), tXrSFB);
        sf_empty_mbar[sf_read_stage].arrive();
        sf_read_stage = (sf_read_stage + 1) % KT::SFConfig::SF_Stages;
        sf_phase ^= uint32_t(sf_read_stage == 0);
      }
    };

    copy_sf_s2r();
    ab_full_mbar[ab_read_stage].wait(ab_phase);
    copy_ab_s2r(Int<0>{});

    int32_t sf_pack_count = num_sf_cycles * KT::SFConfig::SF_Stages;
    for (int32_t sf_pack_idx = 0; sf_pack_idx < sf_pack_count - 1; ++sf_pack_idx) {
      cute::for_each(
          cute::make_int_sequence<KT::SFConfig::kNumTileKPerPackSF>{}, [&](auto k_in_sf) {
            cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
              constexpr int k_block_idx = decltype(k_block)::value;
              if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
                copy_ab_s2r(Int<k_block_idx + 1>{});
              } else {
                ab_empty_mbar[ab_read_stage].arrive();
                advance_ab_read_stage();
                ab_full_mbar[ab_read_stage].wait(ab_phase);
                copy_ab_s2r(Int<0>{});
              }
              cute::gemm(
                  mma,
                  make_zip_tensor(tCrA_up(_, _, k_block), tCrSFA_up_frg(_, _, k_block, k_in_sf)),
                  make_zip_tensor(tCrB(_, _, k_block), tCrSFB_frg(_, _, k_block, k_in_sf)),
                  accum_up);
              cute::gemm(mma,
                         make_zip_tensor(tCrA_gate(_, _, k_block),
                                         tCrSFA_gate_frg(_, _, k_block, k_in_sf)),
                         make_zip_tensor(tCrB(_, _, k_block), tCrSFB_frg(_, _, k_block, k_in_sf)),
                         accum_gate);
            });
          });

      copy_sf_s2r();
    }

    cute::for_each(cute::make_int_sequence<KT::SFConfig::kNumTileKPerPackSF>{}, [&](auto k_in_sf) {
      constexpr bool is_last_k_tile =
          decltype(k_in_sf)::value == KT::SFConfig::kNumTileKPerPackSF - 1;
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        constexpr int k_block_idx = decltype(k_block)::value;
        if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
          copy_ab_s2r(Int<k_block_idx + 1>{});
        } else {
          ab_empty_mbar[ab_read_stage].arrive();
          advance_ab_read_stage();
          if constexpr (!is_last_k_tile) {
            ab_full_mbar[ab_read_stage].wait(ab_phase);
            copy_ab_s2r(Int<0>{});
          }
        }
        cute::gemm(
            mma, make_zip_tensor(tCrA_up(_, _, k_block), tCrSFA_up_frg(_, _, k_block, k_in_sf)),
            make_zip_tensor(tCrB(_, _, k_block), tCrSFB_frg(_, _, k_block, k_in_sf)), accum_up);
        cute::gemm(
            mma, make_zip_tensor(tCrA_gate(_, _, k_block), tCrSFA_gate_frg(_, _, k_block, k_in_sf)),
            make_zip_tensor(tCrB(_, _, k_block), tCrSFB_frg(_, _, k_block, k_in_sf)), accum_gate);
      });
    });

    cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
    CUTE_UNROLL
    for (int i = 0; i < size(accum_up); ++i) {
      float gate = accum_gate(i);
      accum_up(i) *= gate / (1.0f + __expf(-gate));
    }
    sm120_common::utils::epi_pred_stg<KT>(params, accum_up, thread_idx, m_offset, m_boundary,
                                          m_block_idx, n_block_idx, store_empty_mbar);
  }

  CUTE_DEVICE
  static void mma(Params const& params, SharedStorage& shared_storage, int32_t num_sf_cycles,
                  int32_t m_offset, int32_t m_boundary, int32_t m_block_idx, int32_t n_block_idx,
                  int32_t expert_idx, uint32_t& sf_phase, uint32_t& ab_phase, int& epi_stage,
                  uint32_t& se_phase) {
    int thread_idx = int(threadIdx.x);
    typename KT::MMAConfig::TiledMma mma;
    auto thr_mma = mma.get_thread_slice(thread_idx);
    auto accum_up = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    auto accum_gate = partition_fragment_C(mma, take<0, 2>(typename KT::TileShape{}));
    clear(accum_up);
    clear(accum_gate);

    auto [ab_full_mbar, ab_empty_mbar, sf_full_mbar, sf_empty_mbar, store_full_mbar,
          store_empty_mbar] = BaseKernel::get_mbarriers(shared_storage);
    (void)store_full_mbar;

    auto sA = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_A.begin()),
                    typename KT::ABLoadConfig::SmemLayoutA{}));
    auto sB_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_up.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto sB_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_B_gate.begin()),
                    typename KT::ABLoadConfig::SmemLayoutB{}));
    auto sSFA = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFA.begin()),
                    typename KT::SFConfig::SmemLayoutSFA{}));
    auto sSFB_up = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_up.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));
    auto sSFB_gate = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.tensors.load.smem_SFB_gate.begin()),
                    typename KT::SFConfig::SmemLayoutSFB{}));

    auto tCrA = thr_mma.partition_fragment_A(sA(_, _, Int<0>{}));
    auto tCrB_up = thr_mma.partition_fragment_B(sB_up(_, _, Int<0>{}));
    auto tCrB_gate = thr_mma.partition_fragment_B(sB_gate(_, _, Int<0>{}));
    constexpr int K_BLOCK_MAX = decltype(size<2>(tCrA))::value;

    auto s2r_copy_A = make_tiled_copy_A(typename KT::ABLoadConfig::SmemCopyAtomA{}, mma);
    auto s2r_thr_copy_A = s2r_copy_A.get_thread_slice(thread_idx);
    auto tXsA = s2r_thr_copy_A.partition_S(sA);
    auto tXrA = s2r_thr_copy_A.retile_D(tCrA);
    auto s2r_copy_B = make_tiled_copy_B(typename KT::ABLoadConfig::SmemCopyAtomB{}, mma);
    auto s2r_thr_copy_B = s2r_copy_B.get_thread_slice(thread_idx);
    auto tXsB_up = s2r_thr_copy_B.partition_S(sB_up);
    auto tXsB_gate = s2r_thr_copy_B.partition_S(sB_gate);
    auto tXrB_up = s2r_thr_copy_B.retile_D(tCrB_up);
    auto tXrB_gate = s2r_thr_copy_B.retile_D(tCrB_gate);

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
    auto tXsSFB_up = s2r_thr_copy_SFB.partition_S(sSFB_up);
    auto tXsSFB_gate = s2r_thr_copy_SFB.partition_S(sSFB_gate);
    auto tCrSFB_up = KT::SFConfig::partition_fragment_SFB(sSFB_up(_, _, Int<0>{}), thr_mma);
    auto tCrSFB_gate = KT::SFConfig::partition_fragment_SFB(sSFB_gate(_, _, Int<0>{}), thr_mma);
    auto tXrSFB_up = s2r_thr_copy_SFB.retile_D(tCrSFB_up);
    auto tXrSFB_gate = s2r_thr_copy_SFB.retile_D(tCrSFB_gate);
    auto tCrSFB_up_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFB_up);
    auto tCrSFB_gate_frg = KT::SFConfig::transform_fragment_for_qmma(tCrSFB_gate);

    int ab_read_stage = 0;
    int sf_read_stage = 0;

    auto copy_ab_s2r = [&](auto k_block) {
      cute::copy(s2r_copy_A, tXsA(_, _, k_block, ab_read_stage), tXrA(_, _, k_block));
      cute::copy(s2r_copy_B, tXsB_up(_, _, k_block, ab_read_stage), tXrB_up(_, _, k_block));
      cute::copy(s2r_copy_B, tXsB_gate(_, _, k_block, ab_read_stage), tXrB_gate(_, _, k_block));
    };
    auto advance_ab_read_stage = [&]() {
      ab_read_stage = (ab_read_stage + 1) % KT::AB_Stages;
      ab_phase ^= uint32_t(ab_read_stage == 0);
    };
    auto copy_sf_s2r = [&]() {
      if constexpr (KT::SFConfig::SF_Stages == 1) {
        sf_full_mbar[0].wait(sf_phase);
        cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, Int<0>{}), tXrSFA);
        cute::copy(s2r_copy_SFB, tXsSFB_up(_, _, _, Int<0>{}), tXrSFB_up);
        cute::copy(s2r_copy_SFB, tXsSFB_gate(_, _, _, Int<0>{}), tXrSFB_gate);
        sf_empty_mbar[0].arrive();
        sf_phase ^= 1;
      } else {
        sf_full_mbar[sf_read_stage].wait(sf_phase);
        cute::copy(s2r_copy_SFA, tXsSFA(_, _, _, sf_read_stage), tXrSFA);
        cute::copy(s2r_copy_SFB, tXsSFB_up(_, _, _, sf_read_stage), tXrSFB_up);
        cute::copy(s2r_copy_SFB, tXsSFB_gate(_, _, _, sf_read_stage), tXrSFB_gate);
        sf_empty_mbar[sf_read_stage].arrive();
        sf_read_stage = (sf_read_stage + 1) % KT::SFConfig::SF_Stages;
        sf_phase ^= uint32_t(sf_read_stage == 0);
      }
    };

    copy_sf_s2r();
    ab_full_mbar[ab_read_stage].wait(ab_phase);
    copy_ab_s2r(Int<0>{});

    int32_t sf_pack_count = num_sf_cycles * KT::SFConfig::SF_Stages;
    for (int32_t sf_pack_idx = 0; sf_pack_idx < sf_pack_count - 1; ++sf_pack_idx) {
      cute::for_each(
          cute::make_int_sequence<KT::SFConfig::kNumTileKPerPackSF>{}, [&](auto k_in_sf) {
            cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
              constexpr int k_block_idx = decltype(k_block)::value;
              if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
                copy_ab_s2r(Int<k_block_idx + 1>{});
              } else {
                ab_empty_mbar[ab_read_stage].arrive();
                advance_ab_read_stage();
                ab_full_mbar[ab_read_stage].wait(ab_phase);
                copy_ab_s2r(Int<0>{});
              }
              cute::gemm(
                  mma, make_zip_tensor(tCrA(_, _, k_block), tCrSFA_frg(_, _, k_block, k_in_sf)),
                  make_zip_tensor(tCrB_up(_, _, k_block), tCrSFB_up_frg(_, _, k_block, k_in_sf)),
                  accum_up);
              cute::gemm(mma,
                         make_zip_tensor(tCrA(_, _, k_block), tCrSFA_frg(_, _, k_block, k_in_sf)),
                         make_zip_tensor(tCrB_gate(_, _, k_block),
                                         tCrSFB_gate_frg(_, _, k_block, k_in_sf)),
                         accum_gate);
            });
          });

      copy_sf_s2r();
    }

    cute::for_each(cute::make_int_sequence<KT::SFConfig::kNumTileKPerPackSF>{}, [&](auto k_in_sf) {
      constexpr bool is_last_k_tile =
          decltype(k_in_sf)::value == KT::SFConfig::kNumTileKPerPackSF - 1;
      cute::for_each(cute::make_int_sequence<K_BLOCK_MAX>{}, [&](auto k_block) {
        constexpr int k_block_idx = decltype(k_block)::value;
        if constexpr (k_block_idx + 1 < K_BLOCK_MAX) {
          copy_ab_s2r(Int<k_block_idx + 1>{});
        } else {
          ab_empty_mbar[ab_read_stage].arrive();
          advance_ab_read_stage();
          if constexpr (!is_last_k_tile) {
            ab_full_mbar[ab_read_stage].wait(ab_phase);
            copy_ab_s2r(Int<0>{});
          }
        }
        cute::gemm(mma, make_zip_tensor(tCrA(_, _, k_block), tCrSFA_frg(_, _, k_block, k_in_sf)),
                   make_zip_tensor(tCrB_up(_, _, k_block), tCrSFB_up_frg(_, _, k_block, k_in_sf)),
                   accum_up);
        cute::gemm(
            mma, make_zip_tensor(tCrA(_, _, k_block), tCrSFA_frg(_, _, k_block, k_in_sf)),
            make_zip_tensor(tCrB_gate(_, _, k_block), tCrSFB_gate_frg(_, _, k_block, k_in_sf)),
            accum_gate);
      });
    });

    cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
    cutlass::epilogue::thread::SiLu<float> cutlass_silu;
    CUTE_UNROLL
    for (int i = 0; i < size(accum_up); ++i) {
      accum_up(i) *= cutlass_silu(accum_gate(i));
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
          store_empty_mbar] = BaseKernel::get_mbarriers(shared_storage);
    (void)store_full_mbar;
    if (is_tma_thread) {
      cute::for_each(cute::make_int_sequence<KT::AB_Stages>{}, [&](auto stage) {
        ab_full_mbar[stage].init(1);
        ab_empty_mbar[stage].init(KT::MMAConfig::kNumMathThreads);
      });
      cute::for_each(cute::make_int_sequence<KT::SFConfig::SF_Stages>{}, [&](auto stage) {
        sf_full_mbar[stage].init(1);
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

    int32_t num_sf_cycles = sm120_common::math::ceil_div(
        params.K, int(KT::SFConfig::PACK_NK * KT::SFConfig::SF_Stages));

    if (warp_idx >= KT::MMAConfig::kNumMathWarps) {
      cutlass::arch::warpgroup_reg_dealloc<KT::LoadRegisterRequirement>();
      constexpr int sched_warp_idx = KT::MMAConfig::kNumMathWarps;
      constexpr int ab_warp_idx = sched_warp_idx + 1;
      constexpr int sf_warp_idx = ab_warp_idx + 1;
      constexpr int store_warp_idx = sf_warp_idx + 1;

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
            if constexpr (KT::kSwapAB) {
              load_ab_swap(params, shared_storage, blk_coord, tile.m_offset, num_sf_cycles,
                           ab_phase, store_phase);
            } else {
              load_ab(params, shared_storage, blk_coord, tile.m_offset, num_sf_cycles, ab_phase,
                      store_phase);
            }
          }
        }
      } else if (warp_idx == sf_warp_idx) {
        uint32_t sf_phase = 1;
        uint32_t store_phase = 1;
        sm120_common::MoeSchedConsumer<kNumSchedStages> sched_pipeline{shared_storage.sched,
                                                                       lane_predicate};
        sm120_common::MoeWorkTile tile;
        while (sched_pipeline.get_next_tile(tile)) {
          if (lane_predicate) {
            auto blk_coord = sm120_common::utils::make_blk_coord<KT::kSwapAB>(
                tile.m_block, tile.n_block, tile.group);
            if constexpr (KT::kSwapAB) {
              load_sf_swap(params, shared_storage, blk_coord, tile.m_offset, num_sf_cycles,
                           sf_phase, store_phase);
            } else {
              load_sf(params, shared_storage, blk_coord, tile.m_offset, num_sf_cycles, sf_phase,
                      store_phase);
            }
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
          int store_thread_idx = cutlass::canonical_lane_idx();
          int store_stage = 0;
          uint32_t full_phase = 0;
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
          mma_swap(params, shared_storage, num_sf_cycles, tile.m_offset, tile.m_boundary,
                   get<0>(blk_coord), get<1>(blk_coord), sf_phase, ab_phase);
        } else {
          mma(params, shared_storage, num_sf_cycles, tile.m_offset, tile.m_boundary,
              get<0>(blk_coord), get<1>(blk_coord), get<2>(blk_coord), sf_phase, ab_phase,
              epi_stage, se_phase);
        }
      }
    }
  }
};

}  // namespace sm120_blockscaled
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
