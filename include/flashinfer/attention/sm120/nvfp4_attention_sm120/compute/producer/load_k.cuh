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
using cute::make_smem_ptr;
using cute::make_tensor;
using cute::select;
using cute::shape;

template <typename Traits>
struct KLoader {
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using TileShape_MNK = typename Traits::TileShape_MNK;
  using SmemLayoutK = typename Traits::SmemLayoutK;
  using SmemLayoutSFK = typename Traits::SmemLayoutSFK;
  using SmemLayoutDS = typename Traits::SmemLayoutDS;

  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});
  static constexpr bool BlockMean = Traits::BlockMean;

  template <typename MainloopParams, typename PipelineK, typename PipelineStateK, typename TensorGK,
            typename TensorSK, typename TensorGSFK, typename TensorSSFK, typename TensorGDS,
            typename TensorSDS>
  __device__ __forceinline__ static void load_and_stage(
      const MainloopParams& mainloop_params, PipelineK& pipeline_k,
      PipelineStateK& smem_pipe_write_k, const TensorGK& tKgK, const TensorSK& tKsK,
      const TensorGSFK& tKgSFK, const TensorSSFK& tKsSFK, const TensorGDS& tDSgDS,
      const TensorSDS& tDSsDS, int n_block, uint16_t mcast_mask_kv, bool lane_predicate) {
    if (lane_predicate) {
      pipeline_k.producer_acquire(smem_pipe_write_k);

      copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                           mcast_mask_kv),
           tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));

      copy(mainloop_params.tma_load_SFK.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                             mcast_mask_kv),
           tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));

      copy(mainloop_params.tma_load_DS.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                            mcast_mask_kv),
           tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));

      ++smem_pipe_write_k;
    }
  }

  template <typename MainloopParams, typename SharedStorage>
  __device__ __forceinline__ static auto prepare_tma_tensors(const MainloopParams& mainloop_params,
                                                             SharedStorage& shared_storage,
                                                             int m_block, int bidh, int bidb,
                                                             uint2 cluster_local_block_id) {
    auto sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
    auto sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
    auto sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});

    auto mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.shape_K);
    auto mSFK = mainloop_params.tma_load_SFK.get_tma_tensor(shape(mainloop_params.layout_SFK));
    auto mDS = mainloop_params.tma_load_DS.get_tma_tensor(shape(mainloop_params.layout_DS));

    auto gK = local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));
    auto gSFK =
        local_tile(mSFK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));

    auto gDS = [&] {
      if constexpr (BlockMean) {
        return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}),
                          make_coord(m_block, _));
      } else {
        return local_tile(mDS(_, _, bidh, bidb), select<0, 1>(TileShape_MNK{}),
                          make_coord(_0{}, _));
      }
    }();

    auto block_tma_k = mainloop_params.tma_load_K.get_slice(cluster_local_block_id.x);
    auto tKgK = group_modes<0, 3>(block_tma_k.partition_S(gK));
    auto tKsK = group_modes<0, 3>(block_tma_k.partition_D(sK));

    auto block_tma_sfk = mainloop_params.tma_load_SFK.get_slice(cluster_local_block_id.x);
    auto tKgSFK = group_modes<0, 3>(block_tma_sfk.partition_S(gSFK));
    auto tKsSFK = group_modes<0, 3>(block_tma_sfk.partition_D(sSFK));

    auto block_tma_ds = mainloop_params.tma_load_DS.get_slice(cluster_local_block_id.x);
    auto tDSgDS = group_modes<0, 3>(block_tma_ds.partition_S(gDS));
    auto tDSsDS = group_modes<0, 3>(block_tma_ds.partition_D(sDS));

    return cute::make_tuple(tKgK, tKsK, tKgSFK, tKsSFK, tDSgDS, tDSsDS, block_tma_k, block_tma_sfk,
                            block_tma_ds);
  }

  template <typename MainloopParams>
  __device__ __forceinline__ static void prefetch_tma_descriptors(
      const MainloopParams& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFK.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_DS.get_tma_descriptor());
  }
};

}  // namespace nvfp4_attention
