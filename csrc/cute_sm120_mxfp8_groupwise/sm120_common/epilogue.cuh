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
#include <cstdint>

#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/config.hpp>
#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_common {

using namespace cute;

namespace utils {

template <bool kFlat, bool kSwapAB, int kTileN, bool kPerBatchAB>
CUTE_HOST_DEVICE constexpr bool EnableTmaStore() {
  if constexpr (!kFlat) {
    return false;
  } else {
    if constexpr (kSwapAB && !kPerBatchAB && kTileN > 32) {
      return false;
    } else {
      return true;
    }
  }
}

template <int kTileN>
CUTE_HOST_DEVICE constexpr auto SelectSmemCopyAtomR2S() {
  if constexpr (kTileN >= 32)
    return SM90_U16x8_STSM_T{};
  else if constexpr (kTileN >= 16)
    return SM90_U16x4_STSM_T{};
  else
    return SM90_U16x2_STSM_T{};
}

}  // namespace utils

template <int kTileM, int kTileN, typename ElementD, bool kEnabled = true>
struct Sm120BlockScaledTmaStoreConfig {
  using StrideD = Stride<int64_t, Int<1>, int64_t>;

  static constexpr int kEpiTileM = 32;
  static constexpr int kEpiTileN = kTileN;
  static constexpr int StagesD = kTileM / kEpiTileM;
  using EpilogueTile_MN = Shape<Int<kEpiTileM>, Int<kEpiTileN>>;
  static constexpr int kNumEpiM = kTileM / kEpiTileM;
  static constexpr int kNumEpiN = kTileN / kEpiTileN;

  using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
  using SmemLayoutAtomD = GMMA::Layout_K_SW128_Atom<ElementD>;
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomD{}, make_shape(Int<kEpiTileM>{}, Int<kEpiTileN>{}, Int<StagesD>{}),
      Step<_1, _2, _3>{}));
  using CopyOpR2S = SM90_U32x4_STSM_N;
  using CopyOpS2G = SM90_TMA_STORE;
  using TMA_D =
      decltype(make_tma_copy_C_sm90(CopyOpS2G{},
                                    make_tensor(make_gmem_ptr(static_cast<ElementD*>(nullptr)),
                                                repeat_like(StrideD{}, int64_t(0)), StrideD{}),
                                    take<0, 2>(SmemLayoutD{}), EpilogueTile_MN{}));

  struct SharedStorageTmaStore : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
  };
};

template <int kTileM, int kTileN, typename ElementD>
struct Sm120BlockScaledTmaStoreConfig<kTileM, kTileN, ElementD, /*kEnabled=*/false> {
  struct TMA_D {};
  using StrideD = Stride<int64_t, Int<1>, int64_t>;
  static constexpr int StagesD = 1;
};

template <int kTileM, int kTileN, typename ElementD, bool kEnabled = true>
struct Sm120BlockScaledStagedR2GStoreConfig {
  static_assert((kTileM == 64 || kTileM == 128) && kTileN == 128);

  static constexpr int kEpiTileM = 64;
  static constexpr int kEpiTileN = 32;
  static constexpr int StagesD = 2;
  static constexpr int kNumEpiM = kTileM / kEpiTileM;
  static constexpr int kNumEpiN = kTileN / kEpiTileN;
  static constexpr int kNumStoreThreads = 32;

  using EpilogueTile_MN = Shape<Int<kEpiTileM>, Int<kEpiTileN>>;
  using CopyOpR2S = SM90_U32x2_STSM_N;
  using CopyAtomC = Copy_Atom<CopyOpR2S, cutlass::half_t>;
  using SmemLayoutAtomD = GMMA::Layout_K_SW64_Atom<ElementD>;
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomD{}, make_shape(Int<kEpiTileM>{}, Int<kEpiTileN>{}, Int<StagesD>{}),
      Step<_1, _2, _3>{}));

  using SmemCopyAtomS2R = Copy_Atom<UniversalCopy<uint128_t>, ElementD>;
  using TiledCopyS2R = decltype(make_tiled_copy(
      SmemCopyAtomS2R{}, Layout<Shape<_8, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _8>>{}));
  using GmemCopyAtomR2G = SmemCopyAtomS2R;

  struct SharedStorageStagedR2G : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
  };
};

template <int kTileM, int kTileN, typename ElementD>
struct Sm120BlockScaledStagedR2GStoreConfig<kTileM, kTileN, ElementD, false> {
  static constexpr int StagesD = 1;
  struct SharedStorageStagedR2G {};
};

template <int kTileM, int kTileN, typename ElementD>
struct Sm120BlockScaledR2GStoreConfig {
  static constexpr int kSmemON = kTileN;

  using SmemAtomLayoutO = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_8, Shape<_8, _8>>, Stride<_8, Stride<_1, _64>>>{}));
  using SmemLayoutO =
      decltype(tile_to_shape(SmemAtomLayoutO{}, Shape<Int<kTileM>, Int<kSmemON>>{}));
  using SmemCopyAtomR2S = Copy_Atom<AutoVectorizingCopy, ElementD>;
  using SmemCopyAtomS2R = Copy_Atom<UniversalCopy<uint128_t>, ElementD>;
  using GmemCopyAtomR2G = SmemCopyAtomS2R;
  using TiledCopyS2R = decltype(make_tiled_copy(
      SmemCopyAtomS2R{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));

  struct SharedStorageR2G : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutO>> smem_O;
  };
};

template <int kTileM, int kTileN, typename ElementD, bool kEnabled = true>
struct Sm120BlockScaledSwapABTmaStoreConfig {
  static_assert(kTileM == 128, "SwapAB TmaStoreConfig hardcoded for kTileM=128");

  using StrideD = Stride<Int<1>, int64_t, int64_t>;

  static constexpr int kEpiTileM = kTileM;
  static constexpr int kEpiTileN = kTileN;
  static constexpr int StagesD = 1;
  using EpilogueTile_MN = Shape<Int<kEpiTileM>, Int<kEpiTileN>>;
  static constexpr int kNumEpiM = kTileM / kEpiTileM;
  static constexpr int kNumEpiN = kTileN / kEpiTileN;

  using CopyOpR2S = decltype(utils::SelectSmemCopyAtomR2S<kTileN>());
  using CopyAtomC = Copy_Atom<CopyOpR2S, ElementD>;

  using SmemLayoutAtomD = GMMA::Layout_MN_SW128_Atom<ElementD>;
  using SmemLayoutD = decltype(tile_to_shape(
      SmemLayoutAtomD{}, make_shape(Int<kEpiTileM>{}, Int<kEpiTileN>{}, Int<StagesD>{}),
      Step<_1, _2, _3>{}));

  using CopyOpS2G = SM90_TMA_STORE;
  using TMA_D =
      decltype(make_tma_copy_C_sm90(CopyOpS2G{},
                                    make_tensor(make_gmem_ptr(static_cast<ElementD*>(nullptr)),
                                                repeat_like(StrideD{}, int64_t(0)), StrideD{}),
                                    take<0, 2>(SmemLayoutD{}), EpilogueTile_MN{}));

  struct SharedStorageTmaStore : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutD>> smem_D;
  };
};

template <int kTileM, int kTileN, typename ElementD>
struct Sm120BlockScaledSwapABTmaStoreConfig<kTileM, kTileN, ElementD, /*kEnabled=*/false> {
  struct TMA_D {};
  using StrideD = Stride<Int<1>, int64_t, int64_t>;
  static constexpr int StagesD = 1;
};

template <int kTileM, int kTileN, typename ElementD>
struct Sm120BlockScaledSwapABR2GStoreConfig {
  static_assert(kTileM == 128, "SwapAB R2GStoreConfig hardcoded for kTileM=128");

  using StrideD = Stride<Int<1>, int64_t, int64_t>;

  using SmemAtomLayoutO = GMMA::Layout_MN_SW128_Atom<ElementD>;
  using SmemLayoutO = decltype(tile_to_shape(SmemAtomLayoutO{}, Shape<Int<kTileM>, Int<kTileN>>{}));

  using SmemCopyAtomR2S = Copy_Atom<decltype(utils::SelectSmemCopyAtomR2S<kTileN>()), ElementD>;
  using SmemCopyAtomS2R = Copy_Atom<AutoVectorizingCopy, ElementD>;
  using GmemCopyAtomR2G = SmemCopyAtomS2R;

  using TiledCopyS2R =
      decltype(make_tiled_copy(SmemCopyAtomS2R{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{},
                               Layout<Shape<_4, Int<kTileN / 8>>>{}));

  struct SharedStorageR2G : cute::aligned_struct<128, _0> {
    alignas(1024) cute::ArrayEngine<ElementD, cute::cosize_v<SmemLayoutO>> smem_O;
  };
};

namespace utils {

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

template <typename KT, typename Accum>
CUTE_DEVICE auto convert_accum_to_output_type(Accum const& accum) {
  auto epi = make_fragment_like<typename KT::ElementD>(accum);
  auto accum_frg = recast<cutlass::Array<typename KT::ElementAccum, 2>>(accum);
  auto epi_frg = recast<cutlass::Array<typename KT::ElementD, 2>>(epi);
  cutlass::NumericArrayConverter<typename KT::ElementD, typename KT::ElementAccum, 2> converter;
  cute::for_each(cute::make_int_sequence<cute::size(epi_frg)>{},
                 [&](auto i) { epi_frg(i) = converter(accum_frg(i)); });
  return epi;
}

template <typename KT, typename Params, typename Accum, typename Barrier>
CUTE_DEVICE void epi_pred_stg(Params const& params, Accum const& accum, int thread_idx,
                              int32_t m_offset, int32_t m_boundary, int32_t m_block_idx,
                              int32_t n_block_idx, Barrier* store_empty_mbar) {
  static_assert(KT::kFlat || (!KT::kFlat && KT::kSwapAB),
                "epi_pred_stg supports flat GEMM and SwapAB MoE.");
  typename KT::MMAConfig::TiledMma mma;

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

  int32_t residue_M = [&] {
    if constexpr (KT::kFlat)
      return int32_t(params.M);
    else
      return m_boundary - m_offset;
  }();
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

template <typename KT, typename Params, typename SharedStorage, typename Accum, typename Barrier>
CUTE_DEVICE void epi_staged_r2s(Params const& /*params*/, SharedStorage& shared_storage,
                                Accum const& accum, int thread_idx, int32_t m_offset,
                                int32_t m_boundary, int32_t m_block_idx, int& epi_stage,
                                uint32_t& se_phase, Barrier* store_full_mbar,
                                Barrier* store_empty_mbar) {
  using StoreConfig = typename KT::StagedR2GStoreConfig;
  static_assert(KT::kUseStagedR2G);

  int32_t residue_m = m_boundary - m_offset - m_block_idx * KT::kTileM;
  residue_m = residue_m < 0 ? 0 : residue_m;
  residue_m = residue_m > KT::kTileM ? KT::kTileM : residue_m;
  int valid_epi_m = (residue_m + StoreConfig::kEpiTileM - 1) / StoreConfig::kEpiTileM;

  typename KT::MMAConfig::TiledMma mma;
  auto tiled_copy_C_atom = make_tiled_copy_C_atom(typename StoreConfig::CopyAtomC{}, mma);
  auto tiled_copy_r2s = make_tiled_copy_S(
      cute::Copy_Atom<typename StoreConfig::CopyOpR2S, typename KT::ElementD>{}, tiled_copy_C_atom);
  auto thr_copy_r2s = tiled_copy_r2s.get_slice(thread_idx);

  auto sD_epi_ = make_tensor(make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
                             typename StoreConfig::SmemLayoutD{});
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
  constexpr int kMmaMPerEpiM = StoreConfig::kEpiTileM / kMmaTileM;
  constexpr int kMmaNPerEpiN = StoreConfig::kEpiTileN / kMmaTileN;

  for (int epi_n = 0; epi_n < StoreConfig::kNumEpiN; ++epi_n) {
    for (int epi_m = 0; epi_m < valid_epi_m; ++epi_m) {
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

      store_empty_mbar[epi_stage].wait(se_phase);
      copy(tiled_copy_r2s, tRS_rD, tRS_sD(_, _, _, epi_stage));
      cutlass::arch::NamedBarrier::sync(KT::MMAConfig::kNumMathThreads, 0);
      store_full_mbar[epi_stage].arrive();

      if (++epi_stage == StoreConfig::StagesD) {
        epi_stage = 0;
        se_phase ^= 1;
      }
    }
  }
}

template <typename KT, typename Params, typename SharedStorage, typename Barrier>
CUTE_DEVICE void staged_r2g_store(Params const& params, SharedStorage& shared_storage,
                                  int32_t m_offset, int32_t m_boundary, int32_t m_block_idx,
                                  int32_t n_block_idx, int store_thread_idx, uint32_t& full_phase,
                                  int& store_stage, Barrier* store_full_mbar,
                                  Barrier* store_empty_mbar) {
  using StoreConfig = typename KT::StagedR2GStoreConfig;
  static_assert(KT::kUseStagedR2G);

  auto mD_full =
      cute::make_tensor(cute::make_gmem_ptr(params.ptr_D),
                        cute::take<0, 2>(deduce_d_layout<KT::kSwapAB>(params.M, params.N, 1)));
  auto mD_mn = cute::domain_offset(make_offset_coord<KT::kSwapAB>(m_offset), mD_full);
  auto cta_coord = cute::make_coord(m_block_idx, n_block_idx);
  auto gD = cute::local_tile(mD_mn, make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}),
                             cta_coord);
  auto gD_epi = flat_divide(gD, typename StoreConfig::EpilogueTile_MN{});

  auto cD = cute::make_identity_tensor(
      cute::make_shape(cute::Int<KT::kTileM>{}, cute::Int<KT::kTileN>{}));
  auto cD_epi = flat_divide(cD, typename StoreConfig::EpilogueTile_MN{});
  auto residue = make_residue_coord<KT::kSwapAB>(m_boundary - m_offset, params.N, m_block_idx,
                                                 n_block_idx, KT::kTileM, KT::kTileN);
  int32_t residue_m = m_boundary - m_offset - m_block_idx * KT::kTileM;
  residue_m = residue_m < 0 ? 0 : residue_m;
  residue_m = residue_m > KT::kTileM ? KT::kTileM : residue_m;
  int valid_epi_m = (residue_m + StoreConfig::kEpiTileM - 1) / StoreConfig::kEpiTileM;

  auto sD_ = cute::make_tensor(cute::make_smem_ptr(shared_storage.tensors.store.smem_D.begin()),
                               typename StoreConfig::SmemLayoutD{});
  auto sD = cute::as_position_independent_swizzle_tensor(sD_);

  typename StoreConfig::TiledCopyS2R tiled_copy_s2r;
  auto thr_copy_s2r = tiled_copy_s2r.get_slice(store_thread_idx);

  for (int epi_n = 0; epi_n < StoreConfig::kNumEpiN; ++epi_n) {
    for (int epi_m = 0; epi_m < valid_epi_m; ++epi_m) {
      store_full_mbar[store_stage].wait(full_phase);

      auto tSR_sD = thr_copy_s2r.partition_S(sD(_, _, store_stage));
      auto tSR_rD = cute::make_tensor<typename KT::ElementD>(cute::shape(tSR_sD));
      copy(tiled_copy_s2r, tSR_sD, tSR_rD);
      __syncwarp();
      store_empty_mbar[store_stage].arrive();

      auto tRG_rD = thr_copy_s2r.retile_S(tSR_rD);
      auto tRG_gD = thr_copy_s2r.partition_D(gD_epi(_, _, epi_m, epi_n));
      auto tRG_cD = thr_copy_s2r.partition_D(cD_epi(_, _, epi_m, epi_n));
      auto tRG_pD = cute::make_tensor<bool>(cute::Shape<_1>{});

      CUTE_UNROLL
      for (int m = 0; m < cute::size<1>(tRG_gD); ++m) {
        CUTE_UNROLL
        for (int n = 0; n < cute::size<2>(tRG_gD); ++n) {
          auto vector_gD = tRG_gD(_, m, n);
          constexpr int kVectorElements = decltype(cute::size<0>(tRG_gD))::value;
          auto first_coord = tRG_cD(0, m, n);
          auto last_coord = tRG_cD(kVectorElements - 1, m, n);
          tRG_pD(0) = cute::get<0>(first_coord) < cute::get<0>(residue) &&
                      cute::get<1>(last_coord) < cute::get<1>(residue);
          cute::copy_if(typename StoreConfig::GmemCopyAtomR2G{}, tRG_pD, tRG_rD(_, m, n),
                        vector_gD);
        }
      }

      if (++store_stage == StoreConfig::StagesD) {
        store_stage = 0;
        full_phase ^= 1;
      }
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

}  // namespace sm120_common
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
