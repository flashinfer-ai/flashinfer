/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "../../../cutlass_utils.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "fmha_common.hpp"
#include "fmha_fusion.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template <class Element, class CollectiveMmaQK, class CollectiveMmaPV, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class TensorStorage, class PipelineQ,
          class PipelineK, class PipelineV, class Mask, class TileShape>
struct Sm100FmhaLoadTmaWarpspecialized {
  using TileShapeQK = typename CollectiveMmaQK::TileShape;
  using TileShapePV = typename CollectiveMmaPV::TileShape;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
  static constexpr uint32_t NumStagesQ = PipelineQ::Stages;

  // (N, D, (H_R, H_G))
  using ShapeT = cute::Shape<int32_t, int32_t, cute::Shape<int32_t, int32_t>>;
  // (N, D, (H_R, H_G))
  using StrideQ = cute::Shape<int32_t, _1, cute::Shape<int32_t, int32_t>>;
  using StrideK = cute::Shape<int32_t, _1, cute::Shape<_0, int32_t>>;
  using StrideV = cute::Shape<_1, int32_t, cute::Shape<_0, int32_t>>;
  using LayoutQ = cute::Layout<ShapeT, StrideQ>;
  using LayoutK = cute::Layout<ShapeT, StrideK>;
  using LayoutV = cute::Layout<ShapeT, StrideV>;
  struct Arguments {
    const Element* ptr_Q;
    LayoutQ layout_Q;
    const Element* ptr_K;
    LayoutK layout_K;
    const Element* ptr_V;
    LayoutV layout_V;
  };

  // using ShapeLseT = cute::Shape<int32_t, int32_t>;
  // using StrideLseT = cute::Shape<_1, int64_t>;
  // using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  using ClusterLayout_VMNK =
      decltype(tiled_divide(make_layout(Shape<_1, _1, _1>{}),
                            make_tile(typename CollectiveMmaQK::TiledMma::AtomThrID{})));
  using TMA_Q = typename CollectiveMmaQK::Params::TMA_A;
  using TMA_K = typename CollectiveMmaQK::Params::TMA_B;
  using TMA_V = typename CollectiveMmaPV::Params::TMA_B;

  struct Params {
    TMA_Q tma_load_Q;
    LayoutQ layout_Q;
    TMA_K tma_load_K;
    LayoutK layout_K;
    TMA_V tma_load_V;
    LayoutV layout_V;
  };

  template <class ProblemShape>
  static Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
                                        void* workspace) {
    static_assert(is_variable_length_v<tuple_element_t<0, ProblemShape>>);
    static_assert(is_variable_length_v<tuple_element_t<1, ProblemShape>>);
    auto ptr_Q = args.ptr_Q;
    auto ptr_K = args.ptr_K;
    auto ptr_V = args.ptr_V;
    LayoutQ layout_Q = args.layout_Q;
    LayoutK layout_K = args.layout_K;
    LayoutV layout_V = args.layout_V;

    auto mQ = make_tensor(make_gmem_ptr(ptr_Q), layout_Q);
    auto mK = make_tensor(make_gmem_ptr(ptr_K), layout_K);
    auto mV = make_tensor(make_gmem_ptr(ptr_V), layout_V);

    auto cluster_layout_vmnk =
        tiled_divide(make_layout(Shape<_1, _1, _1>{}),
                     make_tile(typename CollectiveMmaQK::TiledMma::AtomThrID{}));
    TMA_Q tma_load_Q = make_tma_atom_A_sm100<Element>(
        GmemTiledCopyQ{}, mQ, SmemLayoutQ{}(_, _, _, _0{}), TileShapeQK{},
        typename CollectiveMmaQK::TiledMma{}, cluster_layout_vmnk);
    TMA_K tma_load_K = make_tma_atom_B_sm100<Element>(
        GmemTiledCopyKV{}, mK, SmemLayoutK{}(_, _, _, _0{}), TileShapeQK{},
        typename CollectiveMmaQK::TiledMma{}, cluster_layout_vmnk);
    TMA_V tma_load_V = make_tma_atom_B_sm100<Element>(
        GmemTiledCopyKV{}, mV, SmemLayoutV{}(_, _, _, _0{}), TileShapePV{},
        typename CollectiveMmaPV::TiledMma{}, cluster_layout_vmnk);

    return Params{tma_load_Q, layout_Q, tma_load_K, layout_K, tma_load_V, layout_V};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& params) {
    cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
  }

  template <class BlkCoord, class ProblemShape, class ParamsProblemShape>
  CUTLASS_DEVICE void load(BlkCoord const& blk_coord, ProblemShape const& problem_shape,
                           Params const& params, ParamsProblemShape const& params_problem_shape,
                           TensorStorage& storage, PipelineQ& pipeline_q,
                           typename PipelineQ::PipelineState& pipeline_q_producer_state,
                           PipelineK& pipeline_k,
                           typename PipelineK::PipelineState& pipeline_k_producer_state,
                           PipelineV& pipeline_v,
                           typename PipelineV::PipelineState& pipeline_v_producer_state) {
    int qo_tile_idx = get<0>(blk_coord);
    int qo_head_idx = get<2, 0>(blk_coord);
    int batch_idx = get<2, 1>(blk_coord);
    int qo_len = get<0>(problem_shape);
    int kv_len = get<1>(problem_shape);
    int qo_segment_offset = get<0>(params_problem_shape).segment_offsets[batch_idx];
    int kv_segment_offset = get<1>(params_problem_shape).segment_offsets[batch_idx];

    int mask_tile_count = Mask{}.get_trip_count(blk_coord, TileShape{}, problem_shape);

    using X = Underscore;

    // this one is only executed by one thread, no need to elect_one

    // Q1, K1, Q2, V1, K2, V2, K3, V3, ...
    // two pipes: Q and KV
    // from Memory (prod) to TensorCore (cons)

    Tensor mQ = params.tma_load_Q.get_tma_tensor(params.layout_Q.shape());
    Tensor mK = params.tma_load_K.get_tma_tensor(params.layout_K.shape());
    Tensor mV = params.tma_load_V.get_tma_tensor(params.layout_V.shape());

    ThrMMA mma_qk = typename CollectiveMmaQK::TiledMma{}.get_slice(0);
    ThrMMA mma_pv = typename CollectiveMmaPV::TiledMma{}.get_slice(0);
    Tensor sQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});

    auto gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShapeQK{}), qo_head_idx, qo_segment_offset,
                                    qo_len);  // (Q, D, _)
    auto gK = get_local_tile_tensor(mK, select<1, 2>(TileShapeQK{}), qo_head_idx, kv_segment_offset,
                                    kv_len);  // (K, D, _)
    auto gV =
        get_local_tile_t_tensor(mV, select<1, 2>(TileShapePV{}), qo_head_idx, kv_segment_offset,
                                kv_len);  // (K, D, _)

    int warp_idx = cutlass::canonical_warp_idx_sync();
    Tensor tSgQ_qdl = mma_qk.partition_A(gQ);
    Tensor tSgK_kdl = mma_qk.partition_B(gK);
    Tensor tOgV_dkl = mma_pv.partition_B(gV);
    auto [tQgQ, tQsQ] = tma_partition(params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 3>(sQ),
                                      group_modes<0, 3>(tSgQ_qdl));  // (TMA, q), (TMA, PIPE)
    auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{}, group_modes<0, 3>(sK),
                                      group_modes<0, 3>(tSgK_kdl));  // (TMA, k), (TMA, PIPE)
    auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{}, group_modes<0, 3>(sV),
                                      group_modes<0, 3>(tOgV_dkl));  // (TMA, k), (TMA, PIPE)

    // blk_coord in decomposed in terms of TileShape, not TileShapeQK
    // As such, it needs to be transformed as
    // (a,b,c): a -> 2*a (Q0) 2*a+1 (Q1)
    //          b -> 2*a (Ki i even) 2*a+1 (Ki i odd)

    uint32_t lane_predicate = cute::elect_one_sync();

    // Q1
    int q0_index = 2 * get<0>(blk_coord);
    int q1_index = 2 * get<0>(blk_coord) + 1;
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_Q.with(*tma_barrier, 0), tQgQ(_, q0_index),
           tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // K1
    int k_index = 0;
    pipeline_k.producer_acquire(pipeline_k_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_k.producer_get_barrier(pipeline_k_producer_state);
      copy(params.tma_load_K.with(*tma_barrier, 0), tKgK(_, k_index),
           tKsK(_, pipeline_k_producer_state.index()));
    }
    ++pipeline_k_producer_state;
    k_index += 1;

    // Q2
    pipeline_q.producer_acquire(pipeline_q_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_q.producer_get_barrier(pipeline_q_producer_state);
      copy(params.tma_load_Q.with(*tma_barrier, 0), tQgQ(_, q1_index),
           tQsQ(_, pipeline_q_producer_state.index()));
    }
    ++pipeline_q_producer_state;

    // V1
    int v_index = 0;
    pipeline_v.producer_acquire(pipeline_v_producer_state);
    if (lane_predicate) {
      auto tma_barrier = pipeline_v.producer_get_barrier(pipeline_v_producer_state);
      copy(params.tma_load_V.with(*tma_barrier, 0), tVgV(_, v_index),
           tVsV(_, pipeline_v_producer_state.index()));
    }
    ++pipeline_v_producer_state;
    v_index += 1;

    // loop:
    mask_tile_count -= 1;
    for (; mask_tile_count > 0; mask_tile_count -= 1) {
      // Ki
      pipeline_k.producer_acquire(pipeline_k_producer_state);
      if (lane_predicate) {
        auto tma_barrier = pipeline_k.producer_get_barrier(pipeline_k_producer_state);
        copy(params.tma_load_K.with(*tma_barrier, 0), tKgK(_, k_index),
             tKsK(_, pipeline_k_producer_state.index()));
      }
      ++pipeline_k_producer_state;
      k_index += 1;

      // Vi
      pipeline_v.producer_acquire(pipeline_v_producer_state);
      if (lane_predicate) {
        auto tma_barrier = pipeline_v.producer_get_barrier(pipeline_v_producer_state);
        copy(params.tma_load_V.with(*tma_barrier, 0), tVgV(_, v_index),
             tVsV(_, pipeline_v_producer_state.index()));
      }
      ++pipeline_v_producer_state;
      v_index += 1;
    }
  }
};

}  // namespace cutlass::fmha::collective
