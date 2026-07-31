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

namespace nvfp4_attention {

using namespace cute;

template <class Layout>
CUTLASS_DEVICE constexpr auto convert_to_reduction_layout(Layout mma_layout) {
  static_assert(rank(mma_layout) == 3, "Mma Layout should be (MmaAtom, MmaM, MmaN)");
  static_assert(rank(get<0>(shape(mma_layout))) == 2, "MmaAtom should be (AtomN, AtomM)");

  return make_layout(make_layout(get<0, 1>(mma_layout), get<1>(mma_layout)),
                     make_layout(get<0, 0>(mma_layout), get<2>(mma_layout)));
}

template <class Layout>
CUTLASS_DEVICE constexpr auto convert_to_conversion_layout(Layout mma_layout) {
  static_assert(rank(mma_layout) == 3, "Mma Layout should be (MmaAtom, MmaM, MmaN)");
  static_assert(rank(get<0>(shape(mma_layout))) == 2, "MmaAtom should be (AtomN, AtomM)");

  constexpr int MmaAtomN = size<0, 0>(mma_layout);
  constexpr int MmaAtomM = size<0, 1>(mma_layout);
  constexpr int MmaM = size<1>(mma_layout);
  constexpr int MmaN = size<2>(mma_layout);

  static_assert(MmaAtomN % 8 == 0, "MmaAtomN should be multiple of 8.");
  static_assert(MmaAtomM == 2, "MmaAtomM should be 2.");
  static_assert(MmaN % 2 == 0, "MmaN should be multiple of 2.");

  auto mma_n_division = zipped_divide(layout<2>(mma_layout), make_tile(_2{}));
  return make_layout(make_layout(layout<0, 0>(mma_layout),
                                 make_layout(layout<0, 1>(mma_layout), layout<0>(mma_n_division))),
                     layout<1>(mma_layout), layout<1>(mma_n_division));
}

CUTLASS_DEVICE constexpr int qk_acc_col_to_k_col(int col) {
  int const col_in_mma = col & 31;
  int const pair = col_in_mma >> 1;
  int const k_pair = ((pair & 3) << 2) | (pair >> 2);
  return (col & ~31) + (k_pair << 1) + (col_in_mma & 1);
}

}  // namespace nvfp4_attention
