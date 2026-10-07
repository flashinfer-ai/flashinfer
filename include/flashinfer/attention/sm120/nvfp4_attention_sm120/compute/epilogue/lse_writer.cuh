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
#include "cutlass/cutlass.h"

namespace nvfp4_attention {

using namespace cute;

template <typename Traits>
struct LSEWriter {
  using TileShape_MNK = typename Traits::TileShape_MNK;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});
  static constexpr int kNWarps = Traits::kNWarps;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumMmaThreads = kNThreads - cutlass::NumThreadsPerWarpGroup;

  using ShapeLSE = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideLSE = cute::Stride<_1, int64_t, int64_t>;

  // Writes the LSE rows owned by one consumer thread. tiled_mma must be the
  // per-warp-group PV mma whose accumulator softmax_fused reduced over, and
  // row_offset the first sequence position of this warp group's sub-tile
  // (m_block * kBlockM + wg_id * kBlockMPerWG).
  template <typename SoftmaxFused, typename TiledMma, typename Shape, typename Stride>
  __device__ __forceinline__ static void write_lse(float* ptr_LSE, Shape const& shape_LSE,
                                                   Stride const& stride_LSE,
                                                   SoftmaxFused const& softmax_fused,
                                                   float softmax_scale_log2,
                                                   TiledMma const& tiled_mma, int thread_idx,
                                                   int row_offset, int bidh, int bidb) {
    Tensor mLSE = make_tensor(make_gmem_ptr(ptr_LSE), shape_LSE, stride_LSE);

    auto const& row_max = softmax_fused.row_max;
    auto const& row_sum = softmax_fused.row_sum;

    Tensor caccO =
        cute::make_identity_tensor(cute::Shape<Int<Traits::kBlockMPerWG>, Int<kHeadDim>>{});
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor taccOcO = thread_mma.partition_C(caccO);

    // acc fragment atom is (AtomN, AtomM) = (8, 2) for the 16x32 mma
    static_assert(decltype(size<0, 0>(taccOcO))::value % 8 == 0);
    static_assert(decltype(size<0, 1>(taccOcO))::value == 2);

    Tensor taccOcO_row = taccOcO(make_coord(_0{}, _), _, _0{});
    CUTE_STATIC_ASSERT_V(size(row_max) == size(taccOcO_row));

    if (get<1>(taccOcO_row(_0{})) == 0) {
      constexpr float log2_e = 1.44269504088896340736f;
      constexpr float ln_2 = 0.69314718055994530942f;

#pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        const int row = row_offset + get<0>(taccOcO_row(mi));

        if (row < get<0>(shape_LSE)) {
          float max_scaled = row_max(mi) * softmax_scale_log2 / log2_e;
          float sum = row_sum(mi);

          // row_sum carries the 2^-fp8_scalexfp4_scale_log2 factor the
          // softmax bakes into every exp for the FP4 P quantization;
          // remove it so lse is the plain ln-sum-exp of the scaled scores.
          float lse =
              (sum == 0.f || sum != sum)
                  ? INFINITY
                  : (max_scaled + logf(sum) + SoftmaxFused::fp8_scalexfp4_scale_log2 * ln_2);

          mLSE(row, bidh, bidb) = lse;
        }
      }
    }
  }
};

}  // namespace nvfp4_attention
