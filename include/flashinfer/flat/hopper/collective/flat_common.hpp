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
#include "cutlass/kernel_hardware_info.h"

namespace flat::collective {

using namespace cute;

template <typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_reset_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  constexpr int rA = decltype(rank(tA))::value;
  constexpr int rB = decltype(rank(tB))::value;
  constexpr int rC = decltype(rank(tC))::value;
  if constexpr (rA == 2 && rB == 2 && rC == 1) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < size<1>(tA); k_block++) {
      cute::gemm(atom, tA(_, k_block), tB(_, k_block), tC);
      atom.accumulate_ = GMMA::ScaleOut::One;
    }
  } else {
    static_assert(rA == 3 && rB == 3 && rC == 3);
    CUTE_UNROLL
    for (int k_block = 0; k_block < size<2>(tA); k_block++) {
      cute::gemm(atom, tA(_, _, k_block), tB(_, _, k_block), tC);
      atom.accumulate_ = GMMA::ScaleOut::One;
    }
  }
}

template <typename Atom, typename TA, typename TB, typename TC>
CUTE_DEVICE void gemm_zero_acc(Atom& atom, TA const& tA, TB const& tB, TC&& tC) {
  atom.accumulate_ = GMMA::ScaleOut::Zero;
  gemm_reset_zero_acc(atom, tA, tB, tC);
}

template <template <cute::GMMA::Major, cute::GMMA::Major, cute::GMMA::ScaleIn,
                    cute::GMMA::ScaleIn> class Primitive,
          cute::GMMA::Major tA, cute::GMMA::Major tB, cute::GMMA::ScaleIn sA,
          cute::GMMA::ScaleIn sB>
CUTE_DEVICE constexpr auto convert_to_gmma_rs(
    cute::MMA_Atom<Primitive<tA, tB, sA, sB>> const& tiled_mma) {
  using Atom = cute::MMA_Atom<Primitive<tA, tB, sA, sB>>;
  using ElementA = typename Atom::ValTypeA;
  using ElementB = typename Atom::ValTypeB;
  using ElementC = typename Atom::ValTypeC;
  using Shape_MNK = typename Atom::Shape_MNK;
  using RS = decltype(cute::GMMA::rs_op_selector<ElementA, ElementB, ElementC, Shape_MNK, tA, tB,
                                                 sA, sB>());
  return cute::MMA_Atom<RS>{};
}

template <template <cute::GMMA::ScaleIn, cute::GMMA::ScaleIn> class Primitive,
          cute::GMMA::ScaleIn sA, cute::GMMA::ScaleIn sB>
CUTE_DEVICE constexpr auto convert_to_gmma_rs(cute::MMA_Atom<Primitive<sA, sB>> const& tiled_mma) {
  using Atom = cute::MMA_Atom<Primitive<sA, sB>>;
  using ElementA = typename Atom::ValTypeA;
  using ElementB = typename Atom::ValTypeB;
  using ElementC = typename Atom::ValTypeC;
  using Shape_MNK = typename Atom::Shape_MNK;
  constexpr auto tA = cute::GMMA::Major::K;
  constexpr auto tB = cute::GMMA::Major::K;
  using RS = decltype(cute::GMMA::rs_op_selector<ElementA, ElementB, ElementC, Shape_MNK, tA, tB,
                                                 sA, sB>());
  return cute::MMA_Atom<RS>{};
}

template <class Atom, class... Args>
CUTE_DEVICE constexpr auto convert_to_gmma_rs(cute::TiledMMA<Atom, Args...> const& tiled_mma) {
  return cute::TiledMMA<decltype(convert_to_gmma_rs(Atom{})), Args...>{};
}

template <typename CLayout, typename AValueShape>
CUTE_DEVICE constexpr auto convert_c_layout_to_a_layout(CLayout const& c, AValueShape const& a) {
  return make_layout(make_shape(a, shape<1>(c), make_shape(shape<2>(c), size<0>(c) / size(a))),
                     make_stride(stride<0>(c), stride<1>(c),
                                 make_stride(stride<2>(c), size<2>(a) * stride<0, 2>(c))));
}

template <class Layout, class Stages = _1>
CUTE_DEVICE constexpr auto unstage_smem_layout(Layout const& layout, Stages stages = {}) {
  return composition(layout, make_tuple(_, _, make_layout(stages)));
}

template <class Element, class Accumulator, class OperandLayout_TV>
CUTE_DEVICE auto make_acc_into_op(Accumulator const& acc,
                                  OperandLayout_TV const& operand_layout_tv) {
  Tensor operand = make_fragment_like<Element>(
      convert_c_layout_to_a_layout(acc.layout(), shape<1>(operand_layout_tv)));
  Tensor operand_as_acc = make_tensor(operand.data(), acc.layout());

  cute::copy(acc, operand_as_acc);

  if constexpr (sizeof(Element) == 1) {
    // 00 11 22 33 00 11 22 33 acc layout
    // 00 00 11 11 22 22 33 33 operand layout
    // BB AA AA BB AA BB BB AA conflict-free exchange pattern
    //                         16-bit exchange; so process two at a time potentially
    int tid = threadIdx.x % 4;
    auto values_u32 = recast<uint32_t>(operand);

    CUTE_UNROLL
    for (int n = 0; n < size<1>(values_u32); n++) {
      CUTE_UNROLL
      for (int k = 0; k < size<2>(values_u32); k++) {
        CUTE_UNROLL
        for (int ii = 0; ii < 8; ii += 4) {
          uint32_t values_tmp_0 = values_u32(ii / 2 + 0, n, k);
          uint32_t values_tmp_1 = values_u32(ii / 2 + 1, n, k);

          // step A:
          // t 1 v 0 -> t 0 v 1
          // t 2 v 0 -> t 1 v 0
          // t 0 v 1 -> t 2 v 0
          // t 3 v 1 -> t 3 v 1

          int v_to_send = tid == 1 || tid == 2 ? 0 : 1;
          int v_to_recv = v_to_send;
          int t_to_recv_from = (0x3021 >> (tid * 4)) & 0xF;

          uint32_t values_tmp_a = v_to_send == 0 ? values_tmp_0 : values_tmp_1;

          values_tmp_a = __shfl_sync(0xFFFFFFFF, values_tmp_a, t_to_recv_from, 4);

          // step B:
          // t 0 v 0 -> t 0 v 0
          // t 3 v 0 -> t 1 v 1
          // t 1 v 1 -> t 2 v 1
          // t 2 v 1 -> t 3 v 0

          v_to_send = 1 - v_to_send;
          v_to_recv = 1 - v_to_recv;
          t_to_recv_from = (0x2130 >> (tid * 4)) & 0xF;

          uint32_t values_tmp_b = v_to_send == 0 ? values_tmp_0 : values_tmp_1;

          values_tmp_b = __shfl_sync(0xFFFFFFFF, values_tmp_b, t_to_recv_from, 4);

          values_u32(ii / 2 + 0, n, k) =
              __byte_perm(values_tmp_a, values_tmp_b, v_to_send == 0 ? 0x1054 : 0x5410);
          values_u32(ii / 2 + 1, n, k) =
              __byte_perm(values_tmp_a, values_tmp_b, v_to_send == 0 ? 0x3276 : 0x7632);
        }
      }
    }
  }

  return operand;
}

}  // namespace flat::collective
