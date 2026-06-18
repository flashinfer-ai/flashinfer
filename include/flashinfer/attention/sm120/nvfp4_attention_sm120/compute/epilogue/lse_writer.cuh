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

  template <typename SoftmaxFused, typename TiledMma, typename Shape, typename Stride>
  __device__ __forceinline__ static void write_lse(float* ptr_LSE, Shape const& shape_LSE,
                                                   Stride const& stride_LSE,
                                                   SoftmaxFused const& softmax_fused,
                                                   float softmax_scale_log2,
                                                   TiledMma const& tiled_mma, int thread_idx,
                                                   int m_block, int bidh, int bidb) {
    Tensor mLSE = make_tensor(make_gmem_ptr(ptr_LSE), shape_LSE, stride_LSE);
    Tensor gLSE =
        local_tile(mLSE(_, bidh, bidb), cute::Shape<cute::Int<kBlockM>>{}, make_coord(m_block));

    auto const& row_max = softmax_fused.row_max;
    auto const& row_sum = softmax_fused.row_sum;

    Tensor caccO = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor taccOcO = thread_mma.partition_C(caccO);

    static_assert(decltype(size<0, 0>(taccOcO))::value == 2);
    static_assert(decltype(size<0, 1>(taccOcO))::value == 2);

    Tensor taccOcO_row = taccOcO(make_coord(_0{}, _), _, _0{});
    CUTE_STATIC_ASSERT_V(size(row_max) == size(taccOcO_row));

    if (get<1>(taccOcO_row(_0{})) == 0) {
      constexpr float log2_e = 1.44269504088896340736f;
      constexpr float ln_2 = 0.69314718055994530942f;

#pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        const int row = get<0>(taccOcO_row(mi));

        if (row < get<0>(shape_LSE) - m_block * kBlockM) {
          float max_scaled = row_max(mi) * softmax_scale_log2 / log2_e;
          float sum = row_sum(mi);

          float lse = (sum == 0.f || sum != sum) ? INFINITY : (max_scaled + logf(sum));

          gLSE(row) = lse;
        }
      }
    }
  }

  template <typename ShapeO, typename Stride>
  __device__ __forceinline__ static void write_lse_infinity(float* ptr_LSE, ShapeO const& shape_O,
                                                            Stride const& stride_LSE,
                                                            int thread_idx, int m_block, int bidh,
                                                            int bidb) {
    auto shape_LSE = select<0, 2, 3>(shape_O);

    Tensor mLSE = make_tensor(make_gmem_ptr(ptr_LSE), shape_LSE, stride_LSE);
    Tensor gLSE = local_tile(mLSE(_, bidh, bidb), Shape<Int<kBlockM>>{}, make_coord(m_block));

    static_assert(kBlockM <= NumMmaThreads);

    if (thread_idx < get<0>(shape_LSE) - m_block * kBlockM) {
      gLSE(thread_idx) = INFINITY;
    }
  }

  template <typename ShapeO, typename ShapeLSE, typename Stride, typename SoftmaxFused,
            typename TiledMma>
  __device__ __forceinline__ static void run(float* ptr_LSE, ShapeO const& shape_O,
                                             ShapeLSE const& shape_LSE, Stride const& stride_LSE,
                                             SoftmaxFused const& softmax_fused,
                                             float softmax_scale_log2, TiledMma const& tiled_mma,
                                             int thread_idx, int m_block, int bidh, int bidb,
                                             bool is_valid_block) {
    if (is_valid_block) {
      write_lse(ptr_LSE, shape_LSE, stride_LSE, softmax_fused, softmax_scale_log2, tiled_mma,
                thread_idx, m_block, bidh, bidb);
    } else {
      write_lse_infinity(ptr_LSE, shape_O, stride_LSE, thread_idx, m_block, bidh, bidb);
    }
  }
};

__device__ __forceinline__ float compute_lse_log2(float row_max, float row_sum,
                                                  float softmax_scale_log2) {
  constexpr float log2_e = 1.44269504088896340736f;

  if (row_sum == 0.f || row_sum != row_sum) {
    return INFINITY;
  }

  float max_scaled = row_max * softmax_scale_log2;
  float lse_log2 = max_scaled + log2f(row_sum);

  return lse_log2;
}

__device__ __forceinline__ float log2_to_ln(float x_log2) {
  constexpr float ln_2 = 0.69314718055994530942f;
  return x_log2 * ln_2;
}

__device__ __forceinline__ float ln_to_log2(float x_ln) {
  constexpr float log2_e = 1.44269504088896340736f;
  return x_ln * log2_e;
}

}  // namespace nvfp4_attention
