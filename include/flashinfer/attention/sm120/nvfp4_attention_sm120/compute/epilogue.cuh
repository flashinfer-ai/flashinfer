/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>

#include "../primitives/barrier.cuh"
#include "../utils/copy.cuh"
#include "../utils/math.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

namespace nvfp4_attention {

using namespace cute;

template <bool Is_even_MN, bool Is_even_K, bool Clear_OOB_MN, bool Clear_OOB_K, typename TiledCopy,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Engine2,
          typename Layout2, typename Engine3, typename Layout3>
CUTLASS_DEVICE void copy_with_bounds_check(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const& S,
                                           Tensor<Engine1, Layout1>& D,
                                           Tensor<Engine2, Layout2> const& identity_MN,
                                           Tensor<Engine3, Layout3> const& predicate_K,
                                           const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || predicate_K(k)) {
          cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        } else if (Clear_OOB_K) {
          cute::clear(D(_, m, k));
        }
      }
    } else if (Clear_OOB_MN) {
      cute::clear(D(_, m, _));
    }
  }
}

template <typename Ktraits>
struct CollectiveEpilogueFwd {
  using Element = typename Ktraits::ElementOut;
  static constexpr int kBlockM = Ktraits::kBlockM;
  static constexpr int kBlockN = Ktraits::kBlockN;
  static constexpr int kHeadDim = Ktraits::kHeadDim;
  static constexpr int kStoreBlockM = Ktraits::kStoreBlockM;
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using TileShape_O = Shape<Int<kStoreBlockM>, Int<kHeadDim>>;
  static constexpr int kNWarps = Ktraits::kNWarps;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumMmaThreads = kNThreads - cutlass::NumThreadsPerWarpGroup;

  using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static_assert(kHeadDim % kGmemElemsPerLoad == 0,
                "kHeadDim must be a multiple of kGmemElemsPerLoad");
  static constexpr int kGmemThreadsPerRow = kHeadDim / kGmemElemsPerLoad;
  static_assert(NumMmaThreads % kGmemThreadsPerRow == 0,
                "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
  using GmemLayoutAtom =
      Layout<Shape<Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
             Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopyO =
      decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{}, GmemLayoutAtom{},
                               Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

  using SmemLayoutO = typename Ktraits::SmemLayoutO;

  using SmemCopyAtomO = Copy_Atom<SM90_U32x2_STSM_N, Element>;
  using SharedStorage = cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>>;

  using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;
  using StrideLSE = cute::Stride<_1, int64_t, int64_t>;

  using TMA_O = decltype(make_tma_copy(GmemTiledCopyOTMA{},
                                       make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
                                                   repeat_like(StrideO{}, int32_t(0)), StrideO{}),
                                       SmemLayoutO{}, TileShape_O{}, _1{}));

  struct Arguments {
    Element* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    float* ptr_LSE;
    StrideLSE const stride_LSE;
  };

  struct Params {
    Element* ptr_O;
    ShapeO const shape_O;
    StrideO const stride_O;
    float* ptr_LSE;
    StrideLSE const stride_LSE;
    TMA_O tma_store_O;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mO = make_tensor(make_gmem_ptr(args.ptr_O), args.shape_O, args.stride_O);
    TMA_O tma_store_O = make_tma_copy(GmemTiledCopyOTMA{}, mO, SmemLayoutO{}, TileShape_O{}, _1{});
    return {args.ptr_O, args.shape_O, args.stride_O, args.ptr_LSE, args.stride_LSE, tma_store_O};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& epilogue_params) {
    cute::prefetch_tma_descriptor(epilogue_params.tma_store_O.get_tma_descriptor());
  }

  template <typename SharedStorage, typename FrgTensorO, typename TiledMma>
  CUTLASS_DEVICE void mma_store(SharedStorage& shared_storage, TiledMma tiled_mma,
                                FrgTensorO const& tOrO, int thread_idx, int wg_id = 0) {
    using TiledMmaPV_Store = typename Ktraits::TiledMmaPV_Store;
    static constexpr int NumMmaThreads = size(TiledMma{});
    TiledMmaPV_Store tiled_mma_pv_store;
    int consumer_thread_idx_full = thread_idx + wg_id * NumMmaThreads;

    Tensor sO = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_o.begin()), SmemLayoutO{}));
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma_pv_store);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(consumer_thread_idx_full);
    constexpr int numel = decltype(size(tOrO))::value;
    cutlass::NumericArrayConverter<Element, float, numel> convert_op;

    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, numel>*>(tOrO.data()));
    auto tOrO_out = make_tensor(make_rmem_ptr<Element>(&frag), tOrO.layout());
    Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    cutlass::arch::fence_view_async_shared();
  }

  template <typename SharedStorage, typename Params, typename WorkTileInfo,
            typename SchedulerParams>
  CUTLASS_DEVICE void tma_store(SharedStorage& shared_storage, Params const& epilogue_params,
                                WorkTileInfo work_tile_info,
                                SchedulerParams const& scheduler_params, int thread_idx,
                                int store_m_subtile = 0) {
    auto [m_block, bidh, bidb] = work_tile_info.get_block_coord(scheduler_params);
    Tensor sO = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_o.begin()), SmemLayoutO{}));
    Tensor mO = epilogue_params.tma_store_O.get_tma_tensor(epilogue_params.shape_O);
    constexpr int StoreTilesPerMBlock = kBlockM / kStoreBlockM;
    Tensor gO = local_tile(mO(_, _, bidh, bidb), TileShape_O{},
                           make_coord(m_block * StoreTilesPerMBlock + store_m_subtile, _0{}));
    auto block_tma_O = epilogue_params.tma_store_O.get_slice(_0{});
    Tensor tOgO = block_tma_O.partition_D(gO);
    Tensor tOsO = block_tma_O.partition_S(sO);

    cute::copy(epilogue_params.tma_store_O, tOsO, tOgO);
    tma_store_arrive();
  }

  CUTLASS_DEVICE void store_tail() { tma_store_wait<0>(); }

  CUTLASS_DEVICE void store_zero(Params const& epilogue_params, int thread_idx,
                                 cute::tuple<int32_t, int32_t, int32_t> const& block_coord) {
    auto [m_block, bidh, bidb] = block_coord;
    Tensor mO = make_tensor(make_gmem_ptr(epilogue_params.ptr_O), epilogue_params.shape_O,
                            epilogue_params.stride_O);
    Tensor gO =
        local_tile(mO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
    auto shape_LSE = select<0, 2, 3>(epilogue_params.shape_O);
    Tensor mLSE =
        make_tensor(make_gmem_ptr(epilogue_params.ptr_LSE), shape_LSE, epilogue_params.stride_LSE);
    Tensor gLSE = local_tile(mLSE(_, bidh, bidb), Shape<Int<kBlockM>>{}, make_coord(m_block));

    GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(thread_idx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_fragment_like(tOgO);
    clear(tOrO);

    Tensor cO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));

    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
#pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(epilogue_params.shape_O);
    }

    copy_with_bounds_check<false, false, false, false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
        get<0>(epilogue_params.shape_O) - m_block * kBlockM);
    static_assert(kBlockM <= NumMmaThreads);
    if (thread_idx < get<0>(shape_LSE) - m_block * kBlockM) {
      gLSE(thread_idx) = INFINITY;
    }
  }
};

}  // namespace nvfp4_attention
