/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include "flashinfer/flat/cute_ext.hpp"

namespace flat::collective {

using namespace cute;

template <int dim, typename Layout>
constexpr bool is_contiguous(Layout&& layout) {
  auto dim_layout = get<dim>(layout);
  if constexpr (rank(dim_layout) == 0) {
    return stride(dim_layout) == 1;
  } else {
    return stride<0>(dim_layout) == 1;
  }
}

namespace detail::SM80 {

// SM80 version of make_acc_into_op in "flat/hopper/collective/flat_common.hpp"
template <typename CLayout, typename TiledMMA>
CUTE_DEVICE constexpr auto convert_c_layout_to_a_layout(CLayout const& c,
                                                        TiledMMA const& tiled_mma) {
  constexpr auto c_frag_atom_size = size<0>(CLayout{});
  constexpr auto a_frag_atom_size = size<1>(typename TiledMMA::AtomLayoutA_TV{});
  static_assert(a_frag_atom_size % c_frag_atom_size == 0);
  constexpr auto ratio = a_frag_atom_size / c_frag_atom_size;
  if constexpr (ratio == 1) {
    return CLayout{};
  } else {
    // e.g. the mma instruction shape is 16x8x16, we need to convert from ((2,2), MMA_M, MMA_N) to
    // ((2,2,2), MMA_M, MMA_N/2)

    constexpr auto tiler =
        make_shape(_, _, Int<ratio>{});  // keep the first mode (FragAtom) and second mode (MMA_M)
    constexpr auto divided =
        logical_divide(CLayout{}, tiler);  // (FragAtom, MMA_M, (ratio, MMA_N/ratio))

    return make_layout(flatten(make_layout(get<0>(divided), get<2, 0>(divided))), get<1>(divided),
                       get<2, 1>(divided));
  }
}

template <class Element, class Accumulator, class TiledMMA>
CUTE_DEVICE auto make_acc_into_op(Accumulator const& acc, TiledMMA const& tiled_mma) {
  Tensor operand =
      make_fragment_like<Element>(convert_c_layout_to_a_layout(acc.layout(), tiled_mma));
  Tensor operand_as_acc = make_tensor(operand.data(), acc.layout());
  cute::copy(acc, operand_as_acc);
  return operand;
}

}  // namespace detail::SM80

template <class Element, bool GarbageFilledDiagonal, bool GarbageFilledUpperTriangular>
struct CollectiveInverse {
  // FIXME: precision is not good due to half
  static_assert(std::is_same_v<Element, half> || std::is_same_v<Element, cutlass::half_t>,
                "only half is implemented");

  CUTE_DEVICE
  CollectiveInverse(int wg_sync_named_barrier_id)
      : wg_sync_named_barrier_id_(wg_sync_named_barrier_id) {}

  template <typename TensorT>
  CUTE_DEVICE void compute(TensorT&& sT) {
    constexpr auto L =
        typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 64);
    static_assert(size<1>(L) == 64);

    int thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

    if (thread_idx < 64) {  // compute 8x8 inverse on diagnal directly
      auto t8X8sT = flat_divide(sT, Shape<_8, _8>{});
      compute_diagonal_inverse_NxN<8>(t8X8sT(_, _, thread_idx / 8, thread_idx / 8), thread_idx % 8);
    }

    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup,
                                                 wg_sync_named_barrier_id_);

    auto t16X16sT = flat_divide(sT, Shape<_16, _16>{});
    blockwise_diagonal_inversed_8x8_to_16x16(t16X16sT(_, _, thread_idx / 32, thread_idx / 32));

    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup,
                                                 wg_sync_named_barrier_id_);

    if (thread_idx < 64) {
      auto t32X32sT = flat_divide(sT, Shape<_32, _32>{});
      blockwise_diagonal_inversed_16x16_to_32x32(t32X32sT(_, _, thread_idx / 32, thread_idx / 32));
    }
    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup,
                                                 wg_sync_named_barrier_id_);
    blockwise_diagonal_inversed_32x32_to_64x64(sT);
  }

 private:
  template <int N, typename TensorT>
  CUTE_DEVICE void compute_diagonal_inverse_NxN(TensorT&& mat,
                                                int tid_in_group) {  // group_size = N
    constexpr auto L =
        typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == N);
    static_assert(size<1>(L) == N);

    using ElementCompute = float;

    using CopyOp =
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<sizeof(Element) * 8 * N>, Element>;

    auto load_row = [&](int y) {
      auto row = make_tensor<Element>(Shape<Int<N>>{});
      copy(CopyOp{}, std::forward<TensorT>(mat)(y, _), row);

      auto row_cvt = make_tensor_like<ElementCompute>(row);
      copy(row, row_cvt);

      if constexpr (GarbageFilledDiagonal || GarbageFilledUpperTriangular) {
        CUTE_UNROLL
        for (int i = 0; i < N; ++i) {
          row_cvt(i) = i == y ? 1.0f : (i > y ? 0.0f : row_cvt(i));
        }
      }
      return row_cvt;
    };

    auto store_row = [&](int y, auto row) {
      auto row_cvt = make_tensor_like<Element>(row);
      copy(row, row_cvt);
      copy(CopyOp{}, row_cvt, std::forward<TensorT>(mat)(y, _));
    };

    auto row = load_row(tid_in_group);
#define LOAD(y, x) __shfl_sync(0xffffffff, row(x), y, N)

    CUTE_UNROLL
    for (int src_row = 0; src_row < N - 1; ++src_row) {  // idx of src row to eliminate
      auto row_scale = -row(src_row);                    // scale the src row
      CUTE_UNROLL
      for (int i = 0; i < src_row; ++i) {
        auto src_row_value = LOAD(src_row, i);
        row(i) = tid_in_group > src_row ? row_scale * src_row_value + row(i) : row(i);
      }
      row(src_row) = tid_in_group > src_row ? row_scale : row(src_row);
    }

#undef LOAD

    store_row(tid_in_group, row);
  }

  /*
  blockwise inverse has relation as follows
  inv(| A 0 |)     = |          inv(A)       0  |
      | C D |        | -inv(D)C inv(A)   inv(D) |
  */

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_4x4_to_8x8(TensorT&& mat) {
    constexpr auto L =
        typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 8);
    static_assert(size<1>(L) == 8);
    auto mat_NxN_2x2 = flat_divide(std::forward<TensorT>(mat), Shape<_4, _4>{});

    // FIXME: implement
  }

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_8x8_to_16x16(TensorT&& mat) {
    constexpr auto L =
        typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 16);
    static_assert(size<1>(L) == 16);

    static_assert(is_contiguous<0>(L) == 1 || is_contiguous<1>(L) == 1);
    constexpr bool is_col_major = is_contiguous<0>(L);

    auto mat_8x8_2x2 = flat_divide(std::forward<TensorT>(mat), Shape<_8, _8>{});
    using MMA = SM80_16x8x8_F32F16F16F32_TN;
    using TiledMMA = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, Shape<_16, _8, _8>{}));

    using CopyOpD_S2R = std::conditional_t<is_col_major, SM75_U16x2_LDSM_T, SM75_U32x1_LDSM_N>;
    using CopyOpC_S2R = std::conditional_t<is_col_major, SM75_U32x1_LDSM_N, SM75_U16x2_LDSM_T>;
    using CopyOpA_S2R = std::conditional_t<is_col_major, SM75_U32x1_LDSM_N, SM75_U16x2_LDSM_T>;
#ifdef CUTE_ARCH_STSM_SM90_ENABLED
    using CopyOpO_R2S = std::conditional_t<is_col_major, SM90_U16x2_STSM_T, SM90_U32x1_STSM_N>;
#else
    using CopyOpO_R2S = UniversalCopy<Element, Element>;
#endif

    int lane_id = cutlass::canonical_lane_idx();
    auto tiled_mma = TiledMMA{};
    auto thr_mma = tiled_mma.get_thread_slice(lane_id);

    auto D_tiled_copy = make_tiled_copy_A(Copy_Atom<CopyOpD_S2R, Element>{}, tiled_mma);
    auto C_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpC_S2R, Element>{}, tiled_mma);
    auto A_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpA_S2R, Element>{}, tiled_mma);
    auto O_tiled_copy = make_tiled_copy_C(Copy_Atom<CopyOpO_R2S, Element>{}, tiled_mma);

    auto D_thr_copy = D_tiled_copy.get_thread_slice(lane_id);
    auto C_thr_copy = C_tiled_copy.get_thread_slice(lane_id);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(lane_id);
    auto O_thr_copy = O_tiled_copy.get_thread_slice(lane_id);

    Tensor sDinv = mat_8x8_2x2(_, _, _1{}, _1{});
    Tensor sC = select_tensor<1, 0>(mat_8x8_2x2(_, _, _1{}, _0{}));
    Tensor sAinv = select_tensor<1, 0>(mat_8x8_2x2(_, _, _0{}, _0{}));
    Tensor sO = mat_8x8_2x2(_, _, _1{}, _0{});

    Tensor sDinv_m_bcast =
        make_tensor(sDinv.data(), logical_product(sDinv.layout(), Tile<Layout<_2, _0>>{}));
    Tensor sO_m_bcast =
        make_tensor(sO.data(), logical_product(sO.layout(), Tile<Layout<_2, _0>>{}));

    Tensor tOrDinv = make_fragment_like<Element>(partition_shape_A(tiled_mma, Shape<_16, _8>{}));
    Tensor tOrC = thr_mma.partition_fragment_B(sC);
    Tensor tOrAinv = thr_mma.partition_fragment_B(sAinv);

    Tensor tDCrDC = partition_fragment_C(tiled_mma, Shape<_16, _8>{});  // output of -inv(D)C
    Tensor tOrO = partition_fragment_C(tiled_mma, Shape<_16, _8>{});    // output of -inv(D)C inv(A)

    Tensor tOsDinv = D_thr_copy.partition_S(sDinv_m_bcast);
    Tensor tOrDinv_cv = D_thr_copy.retile_D(tOrDinv);
    Tensor tOsC = C_thr_copy.partition_S(sC);
    Tensor tOrC_cv = C_thr_copy.retile_D(tOrC);
    Tensor tOsAinv = A_thr_copy.partition_S(sAinv);
    Tensor tOrAinv_cv = A_thr_copy.retile_D(tOrAinv);
    Tensor tOsO = O_thr_copy.partition_D(sO_m_bcast);
    Tensor tOrO_cv = O_thr_copy.retile_S(tOrO);

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C
    copy(D_tiled_copy, tOsDinv(make_coord(_, _0{}), _, _), tOrDinv_cv(make_coord(_, _0{}), _, _));
    copy(C_tiled_copy, tOsC, tOrC_cv);

    clear(tDCrDC);
    gemm(tiled_mma, tOrDinv, tOrC, tDCrDC);
    transform(tDCrDC(make_coord(_, _0{}), _, _), [](auto v) { return -v; });

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C inv(A)
    Tensor tOrDC = detail::SM80::make_acc_into_op<Element>(tDCrDC, tiled_mma);

    copy(A_tiled_copy, tOsAinv, tOrAinv_cv);
    clear(tOrO);
    gemm(tiled_mma, tOrDC, tOrAinv, tOrO);

    auto tOrO_cv_cvt = make_tensor_like<Element>(tOrO_cv(make_coord(_, _0{}), _, _));
    transform(tOrO_cv(make_coord(_, _0{}), _, _), tOrO_cv_cvt, [](auto v) { return Element(v); });
    copy(O_tiled_copy, tOrO_cv_cvt, tOsO(make_coord(_, _0{}), _, _));
  }

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_16x16_to_32x32(TensorT&& mat) {
    constexpr auto L =
        typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 32);
    static_assert(size<1>(L) == 32);

    static_assert(is_contiguous<0>(L) == 1 || is_contiguous<1>(L) == 1);
    constexpr bool is_col_major = is_contiguous<0>(L);

    using TileShape = Shape<_16, _16, _16>;
    auto mat_16x16_2x2 = flat_divide(std::forward<TensorT>(mat), select<0, 1>(TileShape{}));

    using MMA = SM80_16x8x16_F32F16F16F32_TN;
    using TiledMMA = decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, TileShape{}));

    using CopyOpD_S2R = std::conditional_t<is_col_major, SM75_U16x4_LDSM_T, SM75_U32x2_LDSM_N>;
    using CopyOpC_S2R = std::conditional_t<is_col_major, SM75_U32x2_LDSM_N, SM75_U16x4_LDSM_T>;
    using CopyOpA_S2R = std::conditional_t<is_col_major, SM75_U32x2_LDSM_N, SM75_U16x4_LDSM_T>;
#ifdef CUTE_ARCH_STSM_SM90_ENABLED
    using CopyOpO_R2S = std::conditional_t<is_col_major, SM90_U16x4_STSM_T, SM90_U32x2_STSM_N>;
#else
    using CopyOpO_R2S = UniversalCopy<Element, Element>;
#endif

    int lane_id = cutlass::canonical_lane_idx();
    auto tiled_mma = TiledMMA{};
    auto thr_mma = tiled_mma.get_thread_slice(lane_id);

    auto D_tiled_copy = make_tiled_copy_A(Copy_Atom<CopyOpD_S2R, Element>{}, tiled_mma);
    auto C_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpC_S2R, Element>{}, tiled_mma);
    auto A_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpA_S2R, Element>{}, tiled_mma);
    auto O_tiled_copy = make_tiled_copy_C(Copy_Atom<CopyOpO_R2S, Element>{}, tiled_mma);

    auto D_thr_copy = D_tiled_copy.get_thread_slice(lane_id);
    auto C_thr_copy = C_tiled_copy.get_thread_slice(lane_id);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(lane_id);
    auto O_thr_copy = O_tiled_copy.get_thread_slice(lane_id);

    Tensor sDinv = mat_16x16_2x2(_, _, _1{}, _1{});
    Tensor sC = select_tensor<1, 0>(mat_16x16_2x2(_, _, _1{}, _0{}));
    Tensor sAinv = select_tensor<1, 0>(mat_16x16_2x2(_, _, _0{}, _0{}));
    Tensor sO = mat_16x16_2x2(_, _, _1{}, _0{});

    Tensor tOrDinv = thr_mma.partition_fragment_A(sDinv);
    Tensor tOrC = thr_mma.partition_fragment_B(sC);
    Tensor tOrAinv = thr_mma.partition_fragment_B(sAinv);

    Tensor tDCrDC =
        partition_fragment_C(tiled_mma, select<0, 1>(TileShape{}));  // output of -inv(D)C
    Tensor tOrO =
        partition_fragment_C(tiled_mma, select<0, 1>(TileShape{}));  // output of -inv(D)C inv(A)

    Tensor tOsDinv = D_thr_copy.partition_S(sDinv);
    Tensor tOrDinv_cv = D_thr_copy.retile_D(tOrDinv);
    Tensor tOsC = C_thr_copy.partition_S(sC);
    Tensor tOrC_cv = C_thr_copy.retile_D(tOrC);
    Tensor tOsAinv = A_thr_copy.partition_S(sAinv);
    Tensor tOrAinv_cv = A_thr_copy.retile_D(tOrAinv);
    Tensor tOsO = O_thr_copy.partition_D(sO);
    Tensor tOrO_cv = O_thr_copy.retile_S(tOrO);

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C
    copy(D_tiled_copy, tOsDinv, tOrDinv_cv);
    copy(C_tiled_copy, tOsC, tOrC_cv);

    clear(tDCrDC);
    gemm(tiled_mma, tOrDinv, tOrC, tDCrDC);
    transform(tDCrDC, [](auto v) { return -v; });

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C inv(A)
    Tensor tOrDC = detail::SM80::make_acc_into_op<Element>(tDCrDC, tiled_mma);

    copy(A_tiled_copy, tOsAinv, tOrAinv_cv);
    clear(tOrO);
    gemm(tiled_mma, tOrDC, tOrAinv, tOrO);

    auto tOrO_cv_cvt = make_tensor_like<Element>(tOrO_cv);
    transform(tOrO_cv, tOrO_cv_cvt, [](auto v) { return Element(v); });
    copy(O_tiled_copy, tOrO_cv_cvt, tOsO);
  }

  template <typename TensorT>
  CUTE_DEVICE void blockwise_diagonal_inversed_32x32_to_64x64(TensorT&& mat) {
    constexpr auto L =
        typename std::remove_const_t<std::remove_reference_t<TensorT>>::layout_type{};
    static_assert(rank(L) == 2);
    static_assert(size<0>(L) == 64);
    static_assert(size<1>(L) == 64);

    static_assert(is_contiguous<0>(L) == 1 || is_contiguous<1>(L) == 1);
    constexpr bool is_col_major = is_contiguous<0>(L);

    auto mat_32x32_2x2 = flat_divide(std::forward<TensorT>(mat), select<0, 1>(Shape<_32, _32>{}));
    auto mat_16x2X16x2_2x2 = logical_divide(mat_32x32_2x2, Shape<_16, _16>{});

    using MMA = SM80_16x8x16_F32F16F16F32_TN;
    using TiledMMA1 =
        decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, Shape<_16, _16, _32>{}));
    using TiledMMA2 =
        decltype(make_tiled_mma(MMA{}, Layout<Shape<_1, _1>>{}, Shape<_16, _32, _16>{}));

    using CopyOpD_S2R = std::conditional_t<is_col_major, SM75_U16x8_LDSM_T, SM75_U32x4_LDSM_N>;
    using CopyOpC_S2R = std::conditional_t<is_col_major, SM75_U32x4_LDSM_N, SM75_U16x8_LDSM_T>;
    using CopyOpA_S2R = std::conditional_t<is_col_major, SM75_U32x2_LDSM_N, SM75_U16x4_LDSM_T>;
    using CopyOpO_S2R = std::conditional_t<is_col_major, SM75_U16x8_LDSM_T, SM75_U32x4_LDSM_N>;
    using CopyOpO_S2R = std::conditional_t<is_col_major, SM75_U16x8_LDSM_T, SM75_U32x4_LDSM_N>;
#ifdef CUTE_ARCH_STSM_SM90_ENABLED
    using CopyOpO_R2S = std::conditional_t<is_col_major, SM90_U16x8_STSM_T, SM90_U32x4_STSM_N>;
#else
    using CopyOpO_R2S = UniversalCopy<Element, Element>;
#endif

    int warp_id_in_wg = cutlass::canonical_warp_idx() -
                        cutlass::NumWarpsPerWarpGroup * cutlass::canonical_warp_group_idx();
    int x = warp_id_in_wg / 2;
    int y = warp_id_in_wg % 2;

    int lane_id = cutlass::canonical_lane_idx();
    auto tiled_mma1 = TiledMMA1{};
    auto thr_mma1 = tiled_mma1.get_thread_slice(lane_id);

    auto tiled_mma2 = TiledMMA2{};
    auto thr_mma2 = tiled_mma2.get_thread_slice(lane_id);

    auto D_tiled_copy = make_tiled_copy_A(Copy_Atom<CopyOpD_S2R, Element>{}, tiled_mma1);
    auto C_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpC_S2R, Element>{}, tiled_mma1);
    auto A_tiled_copy = make_tiled_copy_B(Copy_Atom<CopyOpA_S2R, Element>{}, tiled_mma2);
    auto O_tiled_s2r = make_tiled_copy_C(Copy_Atom<CopyOpO_S2R, Element>{}, tiled_mma2);
    auto O_tiled_r2s = make_tiled_copy_C(Copy_Atom<CopyOpO_R2S, Element>{}, tiled_mma2);

    auto D_thr_copy = D_tiled_copy.get_thread_slice(lane_id);
    auto C_thr_copy = C_tiled_copy.get_thread_slice(lane_id);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(lane_id);
    auto O_thr_s2r = O_tiled_s2r.get_thread_slice(lane_id);
    auto O_thr_r2s = O_tiled_r2s.get_thread_slice(lane_id);

    Tensor sDinv = mat_16x2X16x2_2x2(make_coord(_, y), _, _1{}, _1{});
    Tensor sC = select_tensor<1, 0>(mat_16x2X16x2_2x2(_, make_coord(_, x), _1{}, _0{}));
    Tensor sAinv =
        select_tensor<1, 0>(mat_16x2X16x2_2x2(make_coord(_, x), _, _0{}, _0{}));  // NOTE: not y!
    Tensor sO = mat_16x2X16x2_2x2(make_coord(_, y), _, _1{}, _0{});  // needs cross-warp reduction

    Tensor tOrDinv = thr_mma1.partition_fragment_A(sDinv);
    Tensor tOrC = thr_mma1.partition_fragment_B(sC);
    Tensor tOrAinv = thr_mma2.partition_fragment_B(sAinv);

    Tensor tDCrDC = partition_fragment_C(tiled_mma1, Shape<_16, _16>{});  // output of -inv(D)C
    Tensor tOrO = partition_fragment_C(tiled_mma2, Shape<_16, _32>{});  // output of -inv(D)C inv(A)

    Tensor tOsDinv = D_thr_copy.partition_S(sDinv);
    Tensor tOrDinv_cv = D_thr_copy.retile_D(tOrDinv);
    Tensor tOsC = C_thr_copy.partition_S(sC);
    Tensor tOrC_cv = C_thr_copy.retile_D(tOrC);
    Tensor tOsAinv = A_thr_copy.partition_S(sAinv);
    Tensor tOrAinv_cv = A_thr_copy.retile_D(tOrAinv);

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C
    copy(D_tiled_copy, tOsDinv, tOrDinv_cv);
    copy(C_tiled_copy, tOsC, tOrC_cv);

    clear(tDCrDC);
    gemm(tiled_mma1, tOrDinv, tOrC, tDCrDC);
    transform(tDCrDC, [](auto v) { return -v; });

    /////////////////////////////////////////////////////////////////////////////
    // -inv(D)C inv(A)
    Tensor tOrDC = detail::SM80::make_acc_into_op<Element>(tDCrDC, tiled_mma2);

    copy(A_tiled_copy, tOsAinv, tOrAinv_cv);
    clear(tOrO);
    gemm(tiled_mma2, tOrDC, tOrAinv, tOrO);

    auto tOrO_cvt = make_tensor_like<Element>(tOrO);
    transform(tOrO, tOrO_cvt, [](auto v) { return Element(v); });

    // ensure tOsC consumed, tOsC and tOsO are the same buffer
    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup,
                                                 wg_sync_named_barrier_id_);

    Tensor tOsO = O_thr_r2s.partition_D(sO);
    Tensor tOrO_cvt_cv = O_thr_r2s.retile_S(tOrO_cvt);
    if (x == 0) {
      copy(O_tiled_r2s, tOrO_cvt_cv, tOsO);
    }
    cutlass::arch::NamedBarrier::arrive_and_wait(cutlass::NumThreadsPerWarpGroup,
                                                 wg_sync_named_barrier_id_);
    if (x == 1) {
      Tensor tOrO_red = make_tensor_like(tOrO_cvt);
      Tensor tOsO_s = O_thr_s2r.partition_S(sO);
      Tensor tOrO_red_cv = O_thr_s2r.retile_D(tOrO_red);
      copy(O_tiled_s2r, tOsO_s, tOrO_red_cv);
      transform(tOrO_cvt, tOrO_red, tOrO_cvt, [](auto a, auto b) { return a + b; });
      copy(O_tiled_r2s, tOrO_cvt_cv, tOsO);
    }
  }

 private:
  int wg_sync_named_barrier_id_;
};

}  // namespace flat::collective
