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

#include "cute/tensor.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

namespace nvfp4_attention {

using namespace cute;

template <typename Traits>
struct OutputWriter {
  using Element = typename Traits::ElementOut;
  using TileShape_MNK = typename Traits::TileShape_MNK;
  using SmemLayoutO = typename Traits::SmemLayoutO;
  using SmemCopyAtomO = typename Traits::SmemCopyAtomO;
  using GmemTiledCopyOTMA = cute::SM90_TMA_STORE;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});
  static constexpr int kNWarps = Traits::kNWarps;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumMmaThreads = kNThreads - cutlass::NumThreadsPerWarpGroup;

  using ShapeO = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideO = cute::Stride<int64_t, _1, int64_t, int64_t>;

  using TMA_O = decltype(make_tma_copy(GmemTiledCopyOTMA{},
                                       make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)),
                                                   repeat_like(StrideO{}, int32_t(0)), StrideO{}),
                                       SmemLayoutO{}, select<0, 2>(TileShape_MNK{}), _1{}));

  template <typename TMA>
  __device__ __forceinline__ static void prefetch_tma_descriptor(TMA const& tma_store_O) {
    cute::prefetch_tma_descriptor(tma_store_O.get_tma_descriptor());
  }

  template <typename SharedStorage, typename TiledMma, typename FrgTensorO>
  __device__ __forceinline__ static void register_to_smem(SharedStorage& shared_storage,
                                                          TiledMma const& tiled_mma,
                                                          FrgTensorO const& tOrO, int thread_idx) {
    Tensor sO = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_o.begin()), SmemLayoutO{}));

    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

    constexpr int numel = decltype(size(tOrO))::value;
    cutlass::NumericArrayConverter<Element, float, numel> convert_op;

    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<float, numel>*>(tOrO.data()));
    auto tOrO_out = make_tensor(make_rmem_ptr<Element>(&frag), tOrO.layout());

    Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    cutlass::arch::fence_view_async_shared();
  }

  template <typename SharedStorage, typename TMA, typename Shape, typename Stride>
  __device__ __forceinline__ static void smem_to_gmem(SharedStorage& shared_storage,
                                                      TMA const& tma_store_O, Shape const& shape_O,
                                                      Stride const& stride_O, int m_block, int bidh,
                                                      int bidb) {
    Tensor sO = cute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_o.begin()), SmemLayoutO{}));

    Tensor mO = tma_store_O.get_tma_tensor(shape_O);
    Tensor gO =
        local_tile(mO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));

    auto block_tma_O = tma_store_O.get_slice(_0{});
    Tensor tOgO = block_tma_O.partition_D(gO);
    Tensor tOsO = block_tma_O.partition_S(sO);

    cute::copy(tma_store_O, tOsO, tOgO);

    tma_store_arrive();
  }

  template <typename SharedStorage, typename TiledMma, typename FrgTensorO, typename TMA,
            typename Shape, typename Stride>
  __device__ __forceinline__ static void run(SharedStorage& shared_storage,
                                             TiledMma const& tiled_mma, FrgTensorO const& tOrO,
                                             TMA const& tma_store_O, Shape const& shape_O,
                                             Stride const& stride_O, int thread_idx, int m_block,
                                             int bidh, int bidb) {
    register_to_smem(shared_storage, tiled_mma, tOrO, thread_idx);

    smem_to_gmem(shared_storage, tma_store_O, shape_O, stride_O, m_block, bidh, bidb);
  }

  __device__ __forceinline__ static void wait_all_stores() { tma_store_wait<0>(); }

  template <typename Shape, typename Stride>
  __device__ __forceinline__ static void store_zero(Element* ptr_O, Shape const& shape_O,
                                                    Stride const& stride_O, int thread_idx,
                                                    int m_block, int bidh, int bidb) {
    Tensor mO = make_tensor(make_gmem_ptr(ptr_O), shape_O, stride_O);
    Tensor gO =
        local_tile(mO(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0,
                  "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kHeadDim / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0,
                  "NumMmaThreads must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = Layout<
        cute::Shape<cute::Int<NumMmaThreads / kGmemThreadsPerRow>, cute::Int<kGmemThreadsPerRow>>,
        cute::Stride<cute::Int<kGmemThreadsPerRow>, cute::_1>>;
    using GmemTiledCopyO =
        decltype(make_tiled_copy(Copy_Atom<DefaultCopy, Element>{}, GmemLayoutAtom{},
                                 Layout<cute::Shape<cute::_1, cute::Int<kGmemElemsPerLoad>>>{}));

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
      tOpO(k) = get<1>(tOcO(_0{}, _0{}, k)) < get<1>(shape_O);
    }

    copy_with_predicate<false, false, false, false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                                    get<0>(shape_O) - m_block * kBlockM);
  }

 private:
  template <bool Is_even_MN, bool Is_even_K, bool Clear_OOB_MN, bool Clear_OOB_K,
            typename TiledCopy, typename Tensor1, typename Tensor2, typename Tensor3,
            typename Tensor4>
  __device__ __forceinline__ static void copy_with_predicate(TiledCopy const& tiled_copy,
                                                             Tensor1 const& src, Tensor2& dst,
                                                             Tensor3 const& coord,
                                                             Tensor4 const& pred, int max_m) {
    CUTE_STATIC_ASSERT_V(rank(src) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(dst) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(src) == size<0>(dst));
    CUTE_STATIC_ASSERT_V(size<1>(src) == size<1>(dst));
    CUTE_STATIC_ASSERT_V(size<2>(src) == size<2>(dst));
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
    for (int m = 0; m < size<1>(src); ++m) {
      if (Is_even_MN || get<0>(coord(0, m, 0)) < max_m) {
#pragma unroll
        for (int k = 0; k < size<2>(src); ++k) {
          if (Is_even_K || pred(k)) {
            cute::copy(tiled_copy, src(_, m, k), dst(_, m, k));
          } else if (Clear_OOB_K) {
            cute::clear(dst(_, m, k));
          }
        }
      } else if (Clear_OOB_MN) {
        cute::clear(dst(_, m, _));
      }
    }
  }
};

}  // namespace nvfp4_attention
