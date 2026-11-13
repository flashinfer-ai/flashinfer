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
struct DeltaSCorrection {
  using TileShape_MNK = typename Traits::TileShape_MNK;
  using SmemLayoutDS = typename Traits::SmemLayoutDS;
  static constexpr bool BlockMean = Traits::BlockMean;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});

  template <typename TensorAcc, typename TensorSDS, typename PipelineStateK>
  __device__ __forceinline__ static void add_delta_s(TensorAcc& acc, TensorSDS const& sDS,
                                                     PipelineStateK const& smem_pipe_read_k) {
    auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
    auto acc_float4 = recast<float4>(acc);

    int quad_id = (threadIdx.x % 4) * 2;

    for (int i = 0; i < 4; i++) {
      auto num = quad_id + i * 8;

      float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
      float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));

      acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;
      acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;
      acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
      acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;
    }
  }

  template <typename TensorSDS, typename PipelineStateK>
  __device__ __forceinline__ static auto make_lambda(TensorSDS const& sDS,
                                                     PipelineStateK const& smem_pipe_read_k) {
    return [&sDS, &smem_pipe_read_k](auto& acc) { add_delta_s(acc, sDS, smem_pipe_read_k); };
  }

  __device__ __forceinline__ static auto make_noop_lambda() {
    return [](auto& acc) {

    };
  }
};

template <bool UseDeltaS, typename Traits, typename TensorSDS, typename PipelineStateK>
__device__ __forceinline__ auto make_delta_s_lambda(TensorSDS const& sDS,
                                                    PipelineStateK const& smem_pipe_read_k) {
  if constexpr (UseDeltaS) {
    return DeltaSCorrection<Traits>::make_lambda(sDS, smem_pipe_read_k);
  } else {
    return DeltaSCorrection<Traits>::make_noop_lambda();
  }
}

}  // namespace nvfp4_attention
