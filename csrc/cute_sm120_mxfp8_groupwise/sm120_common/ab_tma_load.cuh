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
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cute/layout.hpp>

#include <cutlass/numeric_size.h>

#include "cute_sm120_mxfp8_groupwise/sm120_common/scheduler.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_common {

using namespace cute;

namespace utils {

template <int kPerWarpN, typename Element>
CUTE_HOST_DEVICE constexpr auto SelectSmemCopyAtomB() {
  if constexpr (kPerWarpN >= 16)
    return Copy_Atom<SM75_U32x4_LDSM_N, Element>{};
  else if constexpr (kPerWarpN >= 8)
    return Copy_Atom<SM75_U32x2_LDSM_N, Element>{};
  else
    return Copy_Atom<SM75_U32x1_LDSM_N, Element>{};
}

template <int kTileK, typename Element>
CUTE_HOST_DEVICE constexpr auto SelectSmemLayoutAtomK() {
  if constexpr (kTileK == 128)
    return GMMA::Layout_K_SW128_Atom<Element>{};
  else if constexpr (kTileK == 64)
    return GMMA::Layout_K_SW64_Atom<Element>{};
  else
    return GMMA::Layout_K_SW32_Atom<Element>{};
}

}  // namespace utils

template <int kTileM, int kTileN, int kTileK, int kStages, typename ElementA, typename ElementB>
struct Sm120BlockScaledABLoadConfig {
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;

  static constexpr int kNumMathWarpN = (kTileN >= 32) ? 4 : (kTileN / 8);
  static constexpr int kPerWarpN = kTileN / kNumMathWarpN;

  using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
  using SmemCopyAtomB = decltype(utils::SelectSmemCopyAtomB<kPerWarpN, ElementB>());

  using SmemLayoutAtomA = decltype(utils::SelectSmemLayoutAtomK<kTileK, ElementA>());
  using SmemLayoutAtomB = decltype(utils::SelectSmemLayoutAtomK<kTileK, ElementB>());

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{}, make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{}, make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<kStages>{}),
      Step<_1, _2, _3>{}));

  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;

  using TMA_A =
      decltype(make_tma_copy(SM90_TMA_LOAD{},
                             make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)),
                                         repeat_like(StrideA{}, int64_t(0)), StrideA{}),
                             SmemLayoutA{}(_, _, Int<0>{}),
                             make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{}));

  using TMA_B =
      decltype(make_tma_copy(SM90_TMA_LOAD{},
                             make_tensor(make_gmem_ptr(static_cast<ElementB const*>(nullptr)),
                                         repeat_like(StrideB{}, int64_t(0)), StrideB{}),
                             SmemLayoutB{}(_, _, Int<0>{}),
                             make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})), _1{}));

  static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA>));
  static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutB{})) * cute::sizeof_bits_v<ElementB>));
  static constexpr uint32_t TmaABTransactionBytes = TmaTransactionBytesA + TmaTransactionBytesB;
};

namespace utils {

template <typename KT>
static auto make_ab_tma_descriptors(typename KT::ElementA const* ptr_A,
                                    typename KT::ABLoadConfig::StrideA dA,
                                    typename KT::ElementB const* ptr_B,
                                    typename KT::ABLoadConfig::StrideB dB, int M, int N, int K,
                                    int num_experts) {
  constexpr bool kPerBatchAB = KT::kPerBatchAB;
  constexpr bool kSwapAB = KT::kSwapAB;

  auto* a_src = ptr_A;
  auto da_src = dA;
  int a_src_M = kSwapAB ? N : M;
  int a_src_L = kSwapAB ? num_experts : (kPerBatchAB ? num_experts : 1);

  auto* b_src = ptr_B;
  auto db_src = dB;
  int b_src_N = kSwapAB ? M : N;
  int b_src_L = (kSwapAB && !kPerBatchAB) ? 1 : num_experts;

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

  return cute::make_tuple(tma_load_a, tma_load_b);
}

template <typename KT, typename BlkCoord>
CUTE_DEVICE auto tma_ab_partition(typename KT::ABLoadConfig::TMA_A const& tma_load_a,
                                  typename KT::ABLoadConfig::TMA_B const& tma_load_b, int M, int N,
                                  int K, int num_experts, BlkCoord const& blk_coord,
                                  int32_t m_offset) {
  auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
  using X = Underscore;
  constexpr bool kPerBatchAB = KT::kPerBatchAB;
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

}  // namespace utils

}  // namespace sm120_common
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
