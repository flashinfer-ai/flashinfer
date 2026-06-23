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

using cute::_;
using cute::_0;
using cute::copy;
using cute::get;
using cute::group_modes;
using cute::local_tile;
using cute::make_coord;
using cute::make_shape;
using cute::make_smem_ptr;
using cute::make_tensor;
using cute::shape;

template <typename Traits>
struct VLoader {
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using TileShape_MNK = typename Traits::TileShape_MNK;
  using SmemLayoutV = typename Traits::SmemLayoutV;
  using SmemLayoutSFV = typename Traits::SmemLayoutSFV;

  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  template <typename MainloopParams, typename PipelineV, typename PipelineStateV,
            typename TensorGVt, typename TensorSVt, typename TensorGSFVt, typename TensorSSFVt>
  __device__ __forceinline__ static void load_and_stage(
      const MainloopParams& mainloop_params, PipelineV& pipeline_v,
      PipelineStateV& smem_pipe_write_v, const TensorGVt& tVgVt, const TensorSVt& tVsVt,
      const TensorGSFVt& tVgSFVt, const TensorSSFVt& tVsSFVt, int n_block, uint16_t mcast_mask_kv,
      bool lane_predicate) {
    if (lane_predicate) {
      pipeline_v.producer_acquire(smem_pipe_write_v);

      copy(mainloop_params.tma_load_Vt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                            mcast_mask_kv),
           tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));

      copy(mainloop_params.tma_load_SFVt.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                              mcast_mask_kv),
           tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));

      ++smem_pipe_write_v;
    }
  }

  template <typename MainloopParams, typename SharedStorage>
  __device__ __forceinline__ static auto prepare_tma_tensors(const MainloopParams& mainloop_params,
                                                             SharedStorage& shared_storage,
                                                             int bidh, int bidb,
                                                             uint2 cluster_local_block_id) {
    auto sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutV{});
    auto sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFV{});

    auto mVt = mainloop_params.tma_load_Vt.get_tma_tensor(mainloop_params.shape_Vt);
    auto mSFVt = mainloop_params.tma_load_SFVt.get_tma_tensor(shape(mainloop_params.layout_SFVt));

    auto gVt = local_tile(mVt(_, _, bidh, bidb),
                          make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
                          make_coord(_0{}, _));
    auto gSFVt = local_tile(mSFVt(_, _, bidh, bidb),
                            make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{})),
                            make_coord(_0{}, _));

    auto block_tma_vt = mainloop_params.tma_load_Vt.get_slice(cluster_local_block_id.x);
    auto tVgVt = group_modes<0, 3>(block_tma_vt.partition_S(gVt));
    auto tVsVt = group_modes<0, 3>(block_tma_vt.partition_D(sVt));

    auto block_tma_sfvt = mainloop_params.tma_load_SFVt.get_slice(cluster_local_block_id.x);
    auto tVgSFVt = group_modes<0, 3>(block_tma_sfvt.partition_S(gSFVt));
    auto tVsSFVt = group_modes<0, 3>(block_tma_sfvt.partition_D(sSFVt));

    return cute::make_tuple(tVgVt, tVsVt, tVgSFVt, tVsSFVt, block_tma_vt, block_tma_sfvt);
  }

  template <typename MainloopParams>
  __device__ __forceinline__ static void prefetch_tma_descriptors(
      const MainloopParams& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Vt.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFVt.get_tma_descriptor());
  }
};

}  // namespace nvfp4_attention
