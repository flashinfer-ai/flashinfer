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

#include <cutlass/cutlass.h>

#include "cute/tensor.hpp"

namespace nvfp4_attention {

using namespace cute;

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false,
          bool Clear_OOB_K = true, typename TiledCopy, typename Engine0, typename Layout0,
          typename Engine1, typename Layout1, typename Engine2, typename Layout2, typename Engine3,
          typename Layout3>
CUTLASS_DEVICE void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const& S,
                         Tensor<Engine1, Layout1>& D, Tensor<Engine2, Layout2> const& identity_MN,
                         Tensor<Engine3, Layout3> const& predicate_K, const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

  static_assert(!(Clear_OOB_MN && !Clear_OOB_K), "Cannot clear OOB_MN without clearing OOB_K");

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

}  // namespace nvfp4_attention
