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
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/config.hpp>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include <type_traits>

#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_conversion.h"
#include "math.cuh"
#include "scheduler.cuh"

namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled {

using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm;

namespace utils {

template <bool kSwapAB>
CUTE_DEVICE auto make_blk_coord(int32_t sched_m, int32_t sched_n, int32_t expert_idx) {
  if constexpr (kSwapAB) {
    return cute::make_coord(sched_n, sched_m, expert_idx);
  } else {
    return cute::make_coord(sched_m, sched_n, expert_idx);
  }
}

template <bool kSwapAB>
CUTE_DEVICE auto make_stride_d(int32_t N) {
  if constexpr (kSwapAB) {
    return cute::make_stride(cute::_1{}, int64_t(N));
  } else {
    return cute::make_stride(int64_t(N), cute::_1{});
  }
}

template <bool kSwapAB>
CUTE_HOST_DEVICE auto deduce_d_layout(int32_t M, int32_t N, int32_t L) {
  if constexpr (kSwapAB) {
    return cute::make_layout(cute::make_shape(int(N), int(M), L),
                             cute::make_stride(cute::_1{}, int64_t(N), int64_t(M) * N));
  } else {
    return cute::make_layout(cute::make_shape(int(M), int(N), L),
                             cute::make_stride(int64_t(N), cute::_1{}, int64_t(M) * N));
  }
}

template <bool kSwapAB>
CUTE_DEVICE auto make_offset_coord(int32_t m_offset) {
  if constexpr (kSwapAB) {
    return cute::make_coord(0, m_offset);
  } else {
    return cute::make_coord(m_offset, 0);
  }
}

template <bool kSwapAB>
CUTE_DEVICE auto make_residue_coord(int32_t M, int32_t N, int32_t m_block_idx, int32_t n_block_idx,
                                    int32_t kTileM, int32_t kTileN) {
  if constexpr (kSwapAB) {
    return cute::make_coord(N - kTileM * m_block_idx, M - kTileN * n_block_idx);
  } else {
    return cute::make_coord(M - kTileM * m_block_idx, N - kTileN * n_block_idx);
  }
}

template <typename KT>
static auto make_tma_descriptors(typename KT::ElementA* ptr_A,
                                 typename KT::ABLoadConfig::StrideA dA,
                                 typename KT::ElementB* ptr_B,
                                 typename KT::ABLoadConfig::StrideB dB,
                                 typename KT::SFConfig::ElementSFLoad* ptr_SFA,
                                 typename KT::SFConfig::ElementSFLoad* ptr_SFB, int M, int N, int K,
                                 int num_experts) {
  constexpr bool kPerBatchAB =
      KT::kGemmType == GemmType::Batched || KT::kGemmType == GemmType::MGroupedMasked;
  constexpr bool kSwapAB = KT::kSwapAB;

  auto* a_src = ptr_A;
  auto da_src = dA;
  int a_src_M = kSwapAB ? N : M;
  int a_src_L = kSwapAB ? num_experts : (kPerBatchAB ? num_experts : 1);

  auto* b_src = ptr_B;
  auto db_src = dB;
  int b_src_N = kSwapAB ? M : N;
  int b_src_L = (kSwapAB && !kPerBatchAB) ? 1 : num_experts;

  auto* sfa_src = ptr_SFA;
  auto* sfb_src = ptr_SFB;

  auto tensor_A =
      make_tensor(make_gmem_ptr(a_src), make_layout(make_shape(a_src_M, K, a_src_L), da_src));
  typename KT::ABLoadConfig::TMA_A tma_load_a = make_tma_copy(
      SM90_TMA_LOAD{}, tensor_A, typename KT::ABLoadConfig::SmemLayoutA{}(_, _, Int<0>{}),
      make_shape(shape<0>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

  auto tensor_B =
      make_tensor(make_gmem_ptr(b_src), make_layout(make_shape(b_src_N, K, b_src_L), db_src));
  typename KT::ABLoadConfig::TMA_B tma_load_b = make_tma_copy(
      SM90_TMA_LOAD{}, tensor_B, typename KT::ABLoadConfig::SmemLayoutB{}(_, _, Int<0>{}),
      make_shape(shape<1>(typename KT::TileShape{}), shape<2>(typename KT::TileShape{})), _1{});

  auto sfa_shape = make_shape(a_src_M, b_src_N, K, a_src_L);
  auto sfa_layout = KT::SFConfig::deduce_sfa_layout(sfa_shape, num_experts);
  auto tensor_sfa = make_tensor(make_gmem_ptr(sfa_src), sfa_layout);
  typename KT::SFConfig::TMA_SFA tma_load_sfa = make_tma_copy(
      SM90_TMA_LOAD{}, tensor_sfa, typename KT::SFConfig::SmemLayoutSFA{}(_, _, Int<0>{}),
      make_shape(shape<0>(typename KT::SFConfig::ScaleTileShape{}),
                 shape<2>(typename KT::SFConfig::ScaleTileShape{})),
      _1{});

  auto sfb_shape = make_shape(a_src_M, b_src_N, K, b_src_L);
  auto sfb_layout = KT::SFConfig::deduce_sfb_layout(sfb_shape, num_experts);
  auto tensor_sfb = make_tensor(make_gmem_ptr(sfb_src), sfb_layout);
  typename KT::SFConfig::TMA_SFB tma_load_sfb = make_tma_copy(
      SM90_TMA_LOAD{}, tensor_sfb, typename KT::SFConfig::SmemLayoutSFB{}(_, _, Int<0>{}),
      make_shape(shape<1>(typename KT::SFConfig::ScaleTileShape{}),
                 shape<2>(typename KT::SFConfig::ScaleTileShape{})),
      _1{});

  return cute::make_tuple(tma_load_a, tma_load_b, tma_load_sfa, tma_load_sfb);
}

template <typename KT, typename BlkCoord>
CUTE_DEVICE auto tma_ab_partition(typename KT::ABLoadConfig::TMA_A const& tma_load_a,
                                  typename KT::ABLoadConfig::TMA_B const& tma_load_b, int M, int N,
                                  int K, int num_experts, BlkCoord const& blk_coord,
                                  int32_t m_offset) {
  auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
  using X = Underscore;
  constexpr bool kPerBatchAB =
      KT::kGemmType == GemmType::Batched || KT::kGemmType == GemmType::MGroupedMasked;
  constexpr bool kSwapAB = KT::kSwapAB;

  int a_src_M = kSwapAB ? N : M;
  int a_src_L = kSwapAB ? num_experts : (kPerBatchAB ? num_experts : 1);
  int b_src_N = kSwapAB ? M : N;
  int b_src_L = (kSwapAB && !kPerBatchAB) ? 1 : num_experts;
  int32_t a_tile_idx = m_block_idx;
  int32_t a_batch_idx = kSwapAB ? expert_idx : (kPerBatchAB ? expert_idx : 0);
  int32_t b_tile_idx = n_block_idx;
  int32_t b_batch_idx = (kSwapAB && !kPerBatchAB) ? 0 : expert_idx;

  auto mA_full = tma_load_a.get_tma_tensor(make_shape(a_src_M, K, a_src_L));
  auto mB_full = tma_load_b.get_tma_tensor(make_shape(b_src_N, K, b_src_L));

  auto mA = [&] {
    if constexpr (!kSwapAB && !KT::kFlat)
      return cute::domain_offset(make_coord(m_offset, 0, 0), mA_full);
    else
      return mA_full;
  }();
  auto mB = [&] {
    if constexpr (kSwapAB && !KT::kFlat)
      return cute::domain_offset(make_coord(m_offset, 0, 0), mB_full);
    else
      return mB_full;
  }();

  auto gA_mkl = local_tile(mA, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
  auto gB_nkl = local_tile(mB, typename KT::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});

  auto block_tma_a = tma_load_a.get_slice(0);
  auto block_tma_b = tma_load_b.get_slice(0);

  auto gA = gA_mkl(_, _, a_tile_idx, _, a_batch_idx);
  auto gB = gB_nkl(_, _, b_tile_idx, _, b_batch_idx);

  auto tAgA = block_tma_a.partition_S(gA);
  auto tBgB = block_tma_b.partition_S(gB);

  return cute::make_tuple(tAgA, tBgB);
}

template <typename KT, typename BlkCoord>
CUTE_DEVICE auto tma_sf_partition(typename KT::SFConfig::TMA_SFA const& tma_load_sfa,
                                  typename KT::SFConfig::TMA_SFB const& tma_load_sfb, int M, int N,
                                  int K, int num_experts, BlkCoord const& blk_coord,
                                  int32_t m_offset) {
  auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
  using X = Underscore;
  constexpr bool kPerBatchAB =
      KT::kGemmType == GemmType::Batched || KT::kGemmType == GemmType::MGroupedMasked;
  constexpr bool kSwapAB = KT::kSwapAB;

  int sfa_src_M = kSwapAB ? N : M;
  int sfa_src_L = kSwapAB ? num_experts : (kPerBatchAB ? num_experts : 1);
  int sfb_src_N = kSwapAB ? M : N;
  int sfb_src_L = (kSwapAB && !kPerBatchAB) ? 1 : num_experts;
  int32_t sfa_tile_idx = m_block_idx;
  int32_t sfa_batch_idx = kSwapAB ? expert_idx : (kPerBatchAB ? expert_idx : 0);
  int32_t sfb_tile_idx = n_block_idx;
  int32_t sfb_batch_idx = (kSwapAB && !kPerBatchAB) ? 0 : expert_idx;

  auto sfa_shape = make_shape(sfa_src_M, sfb_src_N, K, sfa_src_L);
  auto mSFA_full =
      tma_load_sfa.get_tma_tensor(shape(KT::SFConfig::deduce_sfa_layout(sfa_shape, num_experts)));

  auto sfb_shape = make_shape(sfa_src_M, sfb_src_N, K, sfb_src_L);
  auto mSFB_full =
      tma_load_sfb.get_tma_tensor(shape(KT::SFConfig::deduce_sfb_layout(sfb_shape, num_experts)));

  auto sf_m_offset = [&] {
    if constexpr (KT::kGemmType == GemmType::MGroupedContiguousWithZeroPadding) {
      return math::compute_padded_offset(m_offset, expert_idx);
    } else {
      return m_offset;
    }
  }();
  auto mSFA = [&] {
    if constexpr (!kSwapAB && !KT::kFlat)
      return cute::domain_offset(make_coord(sf_m_offset, 0, 0), mSFA_full);
    else
      return mSFA_full;
  }();
  auto mSFB = [&] {
    if constexpr (kSwapAB && !KT::kFlat)
      return cute::domain_offset(make_coord(sf_m_offset, 0, 0), mSFB_full);
    else
      return mSFB_full;
  }();

  auto gSFA_mkl = local_tile(mSFA, typename KT::SFConfig::ScaleTileShape{}, make_coord(_, _, _),
                             Step<_1, X, _1>{});
  auto gSFB_nkl = local_tile(mSFB, typename KT::SFConfig::ScaleTileShape{}, make_coord(_, _, _),
                             Step<X, _1, _1>{});

  auto block_tma_sfa = tma_load_sfa.get_slice(0);
  auto block_tma_sfb = tma_load_sfb.get_slice(0);

  auto gSFA = gSFA_mkl(_, _, sfa_tile_idx, _, sfa_batch_idx);
  auto gSFB = gSFB_nkl(_, _, sfb_tile_idx, _, sfb_batch_idx);

  auto tAgSFA = block_tma_sfa.partition_S(gSFA);
  auto tBgSFB = block_tma_sfb.partition_S(gSFB);

  return cute::make_tuple(tAgSFA, tBgSFB);
}

template <typename KT, typename Accum>
CUTE_DEVICE auto convert_accum_to_output_type(Accum const& accum) {
  auto epi = make_fragment_like<typename KT::ElementD>(accum);
  auto accum_frg = recast<Array<typename KT::ElementAccum, 2>>(accum);
  auto epi_frg = recast<Array<typename KT::ElementD, 2>>(epi);
  cutlass::NumericArrayConverter<typename KT::ElementD, typename KT::ElementAccum, 2> converter;
  cute::for_each(cute::make_int_sequence<cute::size(epi_frg)>{},
                 [&](auto i) { epi_frg(i) = converter(accum_frg(i)); });
  return epi;
}

template <typename KT, typename Params, typename Accum, typename Barrier>
CUTE_DEVICE void epi_pred_stg(Params const& params, Accum const& accum, int thread_idx,
                              int32_t m_offset, int32_t m_boundary, int32_t m_block_idx,
                              int32_t n_block_idx, Barrier* store_empty_mbar) {
  static_assert(!KT::kFlat && KT::kSwapAB,
                "epi_pred_stg is for SwapAB MoE only (!kFlat && kSwapAB).");
  typename KT::MMAConfig::TiledMma mma;

  auto mD_full = cute::make_tensor(
      cute::make_gmem_ptr(params.ptr_D),
      cute::take<0, 2>(deduce_d_layout<KT::kSwapAB>(params.M, params.N, /*L=*/1)));
  auto mD_mn = cute::domain_offset(make_offset_coord<KT::kSwapAB>(m_offset), mD_full);
  auto cta_coord = cute::make_coord(m_block_idx, n_block_idx);
  auto gD = cute::local_tile(mD_mn, make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}),
                             cta_coord);
  auto cD = cute::make_identity_tensor(
      cute::make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}));

  int32_t residue_M = m_boundary - m_offset;
  auto residue = make_residue_coord<KT::kSwapAB>(residue_M, params.N, m_block_idx, n_block_idx,
                                                 KT::kTileM, KT::kTileN);

  auto thr_mma = mma.get_thread_slice(thread_idx);
  auto tCgD = thr_mma.partition_C(gD);
  auto tCcD = thr_mma.partition_C(cD);
  auto epi = convert_accum_to_output_type<KT>(accum);

  store_empty_mbar[0].arrive();

  CUTE_UNROLL
  for (int i = 0; i < cute::size(epi); ++i) {
    int m_in = cute::get<0>(tCcD(i));
    int n_in = cute::get<1>(tCcD(i));
    if (m_in < cute::get<0>(residue) && n_in < cute::get<1>(residue)) {
      tCgD(i) = epi(i);
    }
  }
}

template <typename KT, typename Params, typename SharedStorage, typename Accum, typename Barrier>
CUTE_DEVICE void epi_pred_r2g(Params const& params, SharedStorage& shared_storage,
                              Accum const& accum, int thread_idx, int32_t m_offset,
                              int32_t m_boundary, int32_t m_block_idx, int32_t n_block_idx,
                              int32_t expert_idx, Barrier* store_empty_mbar) {
  typename KT::MMAConfig::TiledMma mma;
  auto epi = convert_accum_to_output_type<KT>(accum);

  auto sD_ = cute::make_tensor(cute::make_smem_ptr(shared_storage.tensors.store.smem_O.begin()),
                               typename KT::R2GStoreConfig::SmemLayoutO{});
  auto sD = as_position_independent_swizzle_tensor(sD_);
  auto tiled_copy_R2S =
      cute::make_tiled_copy_C(typename KT::R2GStoreConfig::SmemCopyAtomR2S{}, mma);
  auto thr_copy_R2S = tiled_copy_R2S.get_slice(thread_idx);
  auto tRS_rD = thr_copy_R2S.retile_S(epi);
  auto tRS_sD = thr_copy_R2S.partition_D(sD);
  cute::copy(tiled_copy_R2S, tRS_rD, tRS_sD);
  cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);

  typename KT::R2GStoreConfig::TiledCopyS2R tiled_copy_S2R;
  auto thr_copy_S2R = tiled_copy_S2R.get_slice(thread_idx);
  auto tSR_sD = thr_copy_S2R.partition_S(sD);
  auto tSR_rD = cute::make_tensor<typename KT::ElementD>(cute::shape(tSR_sD));

  cute::copy(tiled_copy_S2R, tSR_sD, tSR_rD);
  cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
  store_empty_mbar[0].arrive();

  auto mD_full = cute::make_tensor(
      cute::make_gmem_ptr(params.ptr_D),
      cute::take<0, 2>(deduce_d_layout<KT::kSwapAB>(params.M, params.N, /*L=*/1)));
  auto mD_mn = [&] {
    if constexpr (KT::kFlat)
      return mD_full;
    else
      return cute::domain_offset(make_offset_coord<KT::kSwapAB>(m_offset), mD_full);
  }();
  auto cta_coord = cute::make_coord(m_block_idx, n_block_idx);
  auto gD = cute::local_tile(mD_mn, make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}),
                             cta_coord);
  auto cD = cute::make_identity_tensor(
      cute::make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}));
  auto tRG_rD = thr_copy_S2R.retile_S(tSR_rD);
  auto tRG_gD = thr_copy_S2R.partition_D(gD);
  auto tRG_cD = thr_copy_S2R.partition_D(cD);

  int32_t residue_M = [&] {
    if constexpr (KT::kFlat)
      return int32_t(params.M);
    else
      return m_boundary - m_offset;
  }();
  auto residue = make_residue_coord<KT::kSwapAB>(residue_M, params.N, m_block_idx, n_block_idx,
                                                 KT::kTileM, KT::kTileN);
  CUTE_UNROLL
  for (int m = 0; m < cute::size<1>(tRG_gD); ++m) {
    CUTE_UNROLL
    for (int n = 0; n < cute::size<2>(tRG_gD); ++n) {
      if (cute::get<0>(tRG_cD(0, m, n)) < cute::get<0>(residue) &&
          cute::get<1>(tRG_cD(0, m, n)) < cute::get<1>(residue)) {
        cute::copy(typename KT::R2GStoreConfig::GmemCopyAtomR2G{}, tRG_rD(cute::_, m, n),
                   tRG_gD(cute::_, m, n));
      }
    }
  }
}

template <typename KT, typename Params, typename SharedStorage, typename Accum, typename Barrier>
CUTE_DEVICE void epi_r2s(Params const& /*params*/, SharedStorage& shared_storage,
                         Accum const& accum, int thread_idx, int& epi_stage, uint32_t* se_phase,
                         Barrier* store_full_mbar, Barrier* store_empty_mbar) {
  static_assert(KT::kUseTmaStore, "epi_r2s requires kFlat (TMA store path)");

  typename KT::MMAConfig::TiledMma mma;
  auto tiled_copy_C_atom = make_tiled_copy_C_atom(typename KT::TmaStoreConfig::CopyAtomC{}, mma);
  auto tiled_copy_r2s = make_tiled_copy_S(
      cute::Copy_Atom<typename KT::TmaStoreConfig::CopyOpR2S, typename KT::ElementD>{},
      tiled_copy_C_atom);
  auto thr_copy_r2s = tiled_copy_r2s.get_slice(thread_idx);

  auto sD_epi_ = make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
                             typename KT::TmaStoreConfig::SmemLayoutD{});
  auto sD_epi = cute::as_position_independent_swizzle_tensor(sD_epi_);
  auto tRS_sD = thr_copy_r2s.partition_D(sD_epi);

  auto epi = convert_accum_to_output_type<KT>(accum);
  auto tRS_rEpi = thr_copy_r2s.retile_S(epi);

  Layout tRS_rD_layout = make_layout(take<0, 3>(shape(thr_copy_r2s.partition_S(sD_epi))));
  auto tRS_rD = make_tensor<typename KT::ElementD>(tRS_rD_layout);

  constexpr int kR2S_V = decltype(size<0>(tRS_rEpi))::value;
  constexpr int kMmaTilesM = decltype(size<1>(tRS_rEpi))::value;
  constexpr int kMmaTilesN = decltype(size<2>(tRS_rEpi))::value;
  constexpr int kMmaTileM = KT::kTileM / kMmaTilesM;
  constexpr int kMmaTileN = KT::kTileN / kMmaTilesN;
  constexpr int kMmaMPerEpiM = KT::TmaStoreConfig::kEpiTileM / kMmaTileM;
  constexpr int kMmaNPerEpiN = KT::TmaStoreConfig::kEpiTileN / kMmaTileN;

  for (int epi_n = 0; epi_n < KT::TmaStoreConfig::kNumEpiN; ++epi_n) {
    for (int epi_m = 0; epi_m < KT::TmaStoreConfig::kNumEpiM; ++epi_m) {
      int dst = 0;
      for (int mma_n_in_epi = 0; mma_n_in_epi < kMmaNPerEpiN; ++mma_n_in_epi) {
        int mma_n = epi_n * kMmaNPerEpiN + mma_n_in_epi;
        for (int mma_m_in_epi = 0; mma_m_in_epi < kMmaMPerEpiM; ++mma_m_in_epi) {
          int mma_m = epi_m * kMmaMPerEpiM + mma_m_in_epi;
          for (int v = 0; v < kR2S_V; ++v) {
            tRS_rD(dst++) = tRS_rEpi(v, mma_m, mma_n);
          }
        }
      }

      store_empty_mbar[epi_stage].wait(se_phase[epi_stage]);
      se_phase[epi_stage] ^= 1;

      copy(tiled_copy_r2s, tRS_rD, tRS_sD(_, _, _, epi_stage));
      cute::tma_store_fence();
      cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
      store_full_mbar[epi_stage].arrive();

      epi_stage = (epi_stage + 1) % KT::TmaStoreConfig::StagesD;
    }
  }
}

template <typename KT, typename Params, typename SharedStorage, typename Barrier>
CUTE_DEVICE void tma_store(Params const& params, SharedStorage& shared_storage, int32_t m_block_idx,
                           int32_t n_block_idx, int32_t expert_idx, uint32_t* sf_phase,
                           int& epi_stage, Barrier* store_full_mbar, Barrier* store_empty_mbar) {
  using X = Underscore;

  auto mD_mnl = params.tma_store_d.get_tma_tensor(
      shape(utils::deduce_d_layout<KT::kSwapAB>(params.M, params.N, params.num_experts)));
  auto gD_mnl =
      local_tile(mD_mnl, typename KT::TileShape{}, make_coord(_, _, _), Step<_1, _1, X>{});
  auto gD = gD_mnl(_, _, m_block_idx, n_block_idx, expert_idx);
  auto gD_epi = flat_divide(gD, typename KT::TmaStoreConfig::EpilogueTile_MN{});

  auto block_tma_d = params.tma_store_d.get_slice(Int<0>{});
  auto sD_epi_ = make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
                             typename KT::TmaStoreConfig::SmemLayoutD{});
  auto sD_epi = cute::as_position_independent_swizzle_tensor(sD_epi_);

  int prev_signal_stage = -1;
  for (int epi_n = 0; epi_n < KT::TmaStoreConfig::kNumEpiN; ++epi_n) {
    for (int epi_m = 0; epi_m < KT::TmaStoreConfig::kNumEpiM; ++epi_m) {
      store_full_mbar[epi_stage].wait(sf_phase[epi_stage]);

      auto sD_stage = sD_epi(_, _, epi_stage);
      auto bSG_sD = block_tma_d.partition_S(sD_stage);
      auto bSG_gD = block_tma_d.partition_D(gD_epi(_, _, epi_m, epi_n));
      cute::copy(params.tma_store_d, bSG_sD, bSG_gD);
      cute::tma_store_arrive();

      if (prev_signal_stage >= 0) {
        cute::tma_store_wait<KT::TmaStoreConfig::StagesD - 1>();
        store_empty_mbar[prev_signal_stage].arrive();
        sf_phase[prev_signal_stage] ^= 1;
      }

      prev_signal_stage = epi_stage;
      epi_stage = (epi_stage + 1) % KT::TmaStoreConfig::StagesD;
    }
  }
  cute::tma_store_wait<0>();
  store_empty_mbar[prev_signal_stage].arrive();
  sf_phase[prev_signal_stage] ^= 1;
}

}  // namespace utils

}  // namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled
