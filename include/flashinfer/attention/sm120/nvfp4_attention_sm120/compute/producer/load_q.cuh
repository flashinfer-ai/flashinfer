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
using cute::_1;
using cute::copy;
using cute::get;
using cute::local_tile;
using cute::make_coord;
using cute::make_smem_ptr;
using cute::make_tensor;
using cute::select;
using cute::shape;

template <typename Traits>
struct QLoader {
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using TileShape_MNK = typename Traits::TileShape_MNK;
  using SmemLayoutQ = typename Traits::SmemLayoutQ;
  using SmemLayoutSFQ = typename Traits::SmemLayoutSFQ;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kHeadDim = get<2>(TileShape_MNK{});

  template <typename MainloopParams, typename PipelineQ, typename PipelineStateQ, typename TensorGQ,
            typename TensorSQ, typename TensorGSFQ, typename TensorSSFQ>
  __device__ __forceinline__ static void load_and_stage(
      const MainloopParams& mainloop_params, PipelineQ& pipeline_q,
      PipelineStateQ& smem_pipe_write_q, const TensorGQ& tQgQ, const TensorSQ& tQsQ,
      const TensorGSFQ& tQgSFQ, const TensorSSFQ& tQsSFQ, bool lane_predicate) {
    if (lane_predicate) {
      pipeline_q.producer_acquire(smem_pipe_write_q);

      copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
           tQgQ, tQsQ);

      copy(
          mainloop_params.tma_load_SFQ.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
          tQgSFQ, tQsSFQ);

      ++smem_pipe_write_q;
    }
  }

  template <typename MainloopParams, typename SharedStorage>
  __device__ __forceinline__ static auto prepare_tma_tensors(const MainloopParams& mainloop_params,
                                                             SharedStorage& shared_storage,
                                                             int m_block, int bidh, int bidb) {
    auto sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
    auto sSFQ = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});

    auto mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);
    auto mSFQ = mainloop_params.tma_load_SFQ.get_tma_tensor(shape(mainloop_params.layout_SFQ));

    auto gQ =
        local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
    auto gSFQ = local_tile(mSFQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}),
                           make_coord(m_block, _0{}));

    auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
    auto tQgQ = block_tma_q.partition_S(gQ);
    auto tQsQ = block_tma_q.partition_D(sQ);

    auto block_tma_sfq = mainloop_params.tma_load_SFQ.get_slice(_0{});
    auto tQgSFQ = block_tma_sfq.partition_S(gSFQ);
    auto tQsSFQ = block_tma_sfq.partition_D(sSFQ);

    return cute::make_tuple(tQgQ, tQsQ, tQgSFQ, tQsSFQ, block_tma_q, block_tma_sfq);
  }

  template <typename MainloopParams>
  __device__ __forceinline__ static void prefetch_tma_descriptors(
      const MainloopParams& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_SFQ.get_tma_descriptor());
  }
};

}  // namespace nvfp4_attention
