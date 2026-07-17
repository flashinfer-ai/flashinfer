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

#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_size.h>

#include "cute_sm120_mxfp8_groupwise/sm120_common/math.cuh"
// clang-format on

namespace flashinfer::gemm::mxfp8_cute_sm120 {
namespace sm120_blockscaling {

using namespace cute;

template <int kTileM, int kTileN, int kTileK, int kStages, int kGranM, int kGranN, int kGranK,
          typename ElementScale_>
struct SM120BlockScalingSFConfig {
  using ElementScale = ElementScale_;

  static constexpr int SF_Stages = kStages;
  static constexpr int kTileScaleM = (kTileM + kGranM - 1) / kGranM;
  static constexpr int kTileScaleN = (kTileN + kGranN - 1) / kGranN;
  static constexpr int kTileScaleK = (kTileK + kGranK - 1) / kGranK;
  static constexpr int kNumScaleCopyThreads = 32;
  static constexpr int kNumProducerThreadEvents = kNumScaleCopyThreads;

  static_assert(kTileScaleK == 1);
  static_assert(kTileK == kGranK);

  using ScaleGranularityMNK = Shape<Int<kGranM>, Int<kGranN>, Int<kGranK>>;
  using ScaleTileShape = Shape<Int<kTileScaleM>, Int<kTileScaleN>, Int<kTileScaleK>>;
  using SmemLayoutSFA = Layout<Shape<Int<kTileScaleM>, Int<SF_Stages>>>;
  using SmemLayoutSFB = Layout<Shape<Int<kTileScaleN>, Int<SF_Stages>>>;
  using SmemLayoutTmaSFA = Layout<Shape<Int<kTileScaleM>, Int<kTileScaleK>, Int<SF_Stages>>>;

  using SmemCopyAtomSFA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementScale>, ElementScale>;
  using SmemCopyAtomSFB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementScale>, ElementScale>;

  using SmemLayoutSFAViewAsC =
      Layout<Shape<Shape<Int<kGranM>, Int<kTileScaleM>>, Int<kTileN>, Int<SF_Stages>>,
             Stride<Stride<_0, _1>, _0, Int<kTileScaleM>>>;

  using SmemLayoutSFBViewAsC =
      Layout<Shape<Int<kTileM>, Shape<Int<kGranN>, Int<kTileScaleN>>, Int<SF_Stages>>,
             Stride<_0, Stride<_0, _1>, Int<kTileScaleN>>>;

  CUTE_HOST_DEVICE
  static auto get_tma_aligned_size(const int& x) {
    constexpr int kNumTMAAlignmentBytes = 16;
    CUTE_STATIC_ASSERT(kNumTMAAlignmentBytes % sizeof(ElementScale) == 0,
                       "element_size must be a multiple of 16");
    auto alignment = kNumTMAAlignmentBytes / sizeof(ElementScale);
    return sm120_common::math::align(x, alignment);
  }

  CUTE_HOST_DEVICE
  static auto deduce_sfa_layout(int M, int N, int K, int L) {
    (void)N;
    int64_t scale_m = static_cast<int64_t>(sm120_common::math::ceil_div(M, kGranM));
    int64_t scale_k = static_cast<int64_t>(sm120_common::math::ceil_div(K, kGranK));
    return make_layout(make_shape(scale_m, scale_k, int64_t(L)),
                       make_stride(Int<1>{}, scale_m, scale_m * scale_k));
  }

  CUTE_HOST_DEVICE
  static auto deduce_tma_sfa_layout(int M, int N, int K, int L) {
    (void)N;
    int64_t scale_m =
        static_cast<int64_t>(get_tma_aligned_size(sm120_common::math::ceil_div(M, kGranM)));
    int64_t scale_k = static_cast<int64_t>(sm120_common::math::ceil_div(K, kGranK));
    return make_layout(make_shape(scale_m, scale_k, int64_t(L)),
                       make_stride(Int<1>{}, scale_m, scale_m * scale_k));
  }

  CUTE_HOST_DEVICE
  static auto deduce_sfb_layout(int M, int N, int K, int L) {
    (void)M;
    int64_t scale_n = static_cast<int64_t>(sm120_common::math::ceil_div(N, kGranN));
    int64_t scale_k = static_cast<int64_t>(sm120_common::math::ceil_div(K, kGranK));
    return make_layout(make_shape(scale_n, scale_k, int64_t(L)),
                       make_stride(Int<1>{}, scale_n, scale_n * scale_k));
  }
};

template <typename SFConfig, bool kEnabled = true>
struct SM120BlockScalingSfaTmaLoadConfig {
  using ElementScale = typename SFConfig::ElementScale;
  using StrideSFA = Stride<Int<1>, int64_t, int64_t>;
  using TMA_SFA =
      decltype(make_tma_copy(SM90_TMA_LOAD{},
                             make_tensor(make_gmem_ptr(static_cast<ElementScale const*>(nullptr)),
                                         repeat_like(StrideSFA{}, int64_t(0)), StrideSFA{}),
                             typename SFConfig::SmemLayoutTmaSFA{}(_, _, Int<0>{}),
                             make_shape(shape<0>(typename SFConfig::ScaleTileShape{}),
                                        shape<2>(typename SFConfig::ScaleTileShape{})),
                             _1{}));
  static constexpr uint32_t TmaTransactionBytesSFA = static_cast<uint32_t>(
      cutlass::bits_to_bytes(cosize(take<0, 2>(typename SFConfig::SmemLayoutTmaSFA{})) *
                             cute::sizeof_bits_v<ElementScale>));
};

template <typename SFConfig>
struct SM120BlockScalingSfaTmaLoadConfig<SFConfig, false> {
  struct TMA_SFA {};
};

namespace utils {

template <typename KT>
static auto make_sfa_tma_descriptor(typename KT::ElementScale const* ptr_SFA, int M, int N, int K,
                                    int num_experts) {
  constexpr bool kPerBatchAB = KT::kPerBatchAB;

  int sfa_src_M = M;
  int sfa_src_L = kPerBatchAB ? num_experts : 1;
  if constexpr (KT::kGemmType == sm120_common::GemmType::MGroupedContiguousWithZeroPadding) {
    sfa_src_M = sm120_common::math::compute_padded_offset(M, num_experts);
  }

  auto sfa_layout = KT::SFConfig::deduce_tma_sfa_layout(sfa_src_M, N, K, sfa_src_L);
  auto tensor_sfa = make_tensor(make_gmem_ptr(ptr_SFA), sfa_layout);
  typename KT::SfaTmaLoadConfig::TMA_SFA tma_load_sfa = make_tma_copy(
      SM90_TMA_LOAD{}, tensor_sfa, typename KT::SFConfig::SmemLayoutTmaSFA{}(_, _, Int<0>{}),
      make_shape(shape<0>(typename KT::SFConfig::ScaleTileShape{}),
                 shape<2>(typename KT::SFConfig::ScaleTileShape{})),
      _1{});
  return tma_load_sfa;
}

template <typename KT, typename BlkCoord>
CUTE_DEVICE auto tma_sfa_partition(typename KT::SfaTmaLoadConfig::TMA_SFA const& tma_load_sfa,
                                   int M, int N, int K, int num_experts, BlkCoord const& blk_coord,
                                   int32_t m_offset) {
  auto [m_block_idx, n_block_idx, expert_idx] = blk_coord;
  (void)n_block_idx;
  using X = Underscore;
  constexpr bool kPerBatchAB = KT::kPerBatchAB;

  int sfa_src_M = M;
  int sfa_src_L = kPerBatchAB ? num_experts : 1;
  int32_t sfa_batch_idx = kPerBatchAB ? expert_idx : 0;
  int32_t sf_m_offset = m_offset;
  if constexpr (KT::kGemmType == sm120_common::GemmType::MGroupedContiguousWithZeroPadding) {
    sfa_src_M = sm120_common::math::compute_padded_offset(M, num_experts);
    sf_m_offset = sm120_common::math::compute_padded_offset(m_offset, expert_idx);
  }

  auto sfa_layout = KT::SFConfig::deduce_tma_sfa_layout(sfa_src_M, N, K, sfa_src_L);
  auto mSFA_full = tma_load_sfa.get_tma_tensor(shape(sfa_layout));
  auto mSFA = [&] {
    if constexpr (!KT::kFlat) {
      return domain_offset(make_coord(sf_m_offset / KT::kGranM, 0, 0), mSFA_full);
    } else {
      return mSFA_full;
    }
  }();
  auto gSFA_mkl = local_tile(mSFA, typename KT::SFConfig::ScaleTileShape{}, make_coord(_, _, _),
                             Step<_1, X, _1>{});
  auto gSFA = gSFA_mkl(_, _, m_block_idx, _, sfa_batch_idx);
  auto block_tma_sfa = tma_load_sfa.get_slice(0);
  return block_tma_sfa.partition_S(gSFA);
}

}  // namespace utils

}  // namespace sm120_blockscaling
}  // namespace flashinfer::gemm::mxfp8_cute_sm120
