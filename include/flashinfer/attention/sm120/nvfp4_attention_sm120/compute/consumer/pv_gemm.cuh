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

#include "../../quantization/fp4_convert.cuh"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace nvfp4_attention {

using cute::_;
using cute::_0;
using cute::_1;
using cute::_2;
using cute::_3;
using cute::_4;
using cute::_5;
using cute::_6;
using cute::_7;
using cute::copy;
using cute::get;
using cute::make_coord;
using cute::make_tensor_like;
using cute::make_zip_tensor;
using cute::recast;
using cute::size;
using cute::Tensor;

template <typename Traits>
struct PVGemmComputer {
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using TileShape_MNK = typename Traits::TileShape_MNK;
  using TiledMmaPV = typename Traits::TiledMmaPV;
  using SmemCopyAtomKV = typename Traits::SmemCopyAtomKV;
  using SmemCopyAtomSF = typename Traits::SmemCopyAtomSF;
  using LayoutP = typename Traits::LayoutP;
  using LayoutSFP = typename Traits::LayoutSFP;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kBlockK = get<2>(TileShape_MNK{});

  template <typename SmemTiledCopyV, typename SmemTiledCopySFV, typename TensorSsVt,
            typename TensorSsSFVt, typename TensorSrVt, typename TensorSrSFVt,
            typename PipelineStateV>
  __device__ __forceinline__ static void copy_v_block(
      SmemTiledCopyV const& smem_tiled_copy_V, SmemTiledCopySFV const& smem_tiled_copy_SFV,
      TensorSsVt const& tOsVt, TensorSsSFVt const& tOsSFVt, TensorSrVt& tOrVt_copy_view,
      TensorSrSFVt& tOrSFVt_copy_view, PipelineStateV const& smem_pipe_read_v, auto block_id) {
    auto tOsVt_stage = tOsVt(_, _, _, smem_pipe_read_v.index());
    auto tOsSFVt_stage = tOsSFVt(_, _, _, smem_pipe_read_v.index());

    copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id), tOrVt_copy_view(_, _, block_id));
    copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id), tOrSFVt_copy_view(_, _, block_id));
  }

  template <typename TensorAcc, typename TensorMaxP, typename TensorRP, typename TensorRSFP>
  __device__ __forceinline__ static void quantize_p(TensorAcc const& acc_conversion_view,
                                                    TensorMaxP const& AbsMaxP, TensorRP& tOrP,
                                                    TensorRSFP& tOrSFP, int mma_k) {
    Tensor AbsMaxP_stagek = AbsMaxP(_, make_coord(_, _, mma_k));
    Tensor acc_conversion_stagek = acc_conversion_view(_, _, mma_k);

    Tensor SFP = make_tensor_like<cutlass::float_ue4m3_t>(AbsMaxP_stagek.layout());
    Tensor SFP_uint32_view = recast<uint32_t>(SFP);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(AbsMaxP_stagek); i += 4) {
      uint32_t& tmp = SFP_uint32_view(i / 4);
      nvfp4_attention::packed_float_to_ue4m3(AbsMaxP_stagek(i), AbsMaxP_stagek(i + 1),
                                             AbsMaxP_stagek(i + 2), AbsMaxP_stagek(i + 3), tmp);
    }

    int const quad_id = threadIdx.x & 3;
    uint32_t MASK = (0xFF00FF) << ((quad_id & 1) * 8);
    Tensor tOrSFP_uint32_view = recast<uint32_t>(tOrSFP(_, _, mma_k));
    Tensor tOrP_uint32_view = recast<uint32_t>(tOrP(_, _, mma_k));

    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < size<1>(tOrP); ++mma_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 4; ++i) {
        nvfp4_attention::packed_float_to_e2m1(acc_conversion_stagek(make_coord(_0{}, i), mma_m),
                                              acc_conversion_stagek(make_coord(_1{}, i), mma_m),
                                              acc_conversion_stagek(make_coord(_2{}, i), mma_m),
                                              acc_conversion_stagek(make_coord(_3{}, i), mma_m),
                                              acc_conversion_stagek(make_coord(_4{}, i), mma_m),
                                              acc_conversion_stagek(make_coord(_5{}, i), mma_m),
                                              acc_conversion_stagek(make_coord(_6{}, i), mma_m),
                                              acc_conversion_stagek(make_coord(_7{}, i), mma_m),
                                              tOrP_uint32_view(i, mma_m));
      }

      uint32_t local_sfp = SFP_uint32_view(_0{}, _0{}, mma_m);
      uint32_t peer_sfp = __shfl_xor_sync(int32_t(-1), local_sfp, 2);
      if ((quad_id & 1) == 0) {
        uint32_t sfp = (local_sfp & MASK) | ((peer_sfp & MASK) << 8);
        tOrSFP_uint32_view(_0{}, mma_m) = sfp;
      } else {
        uint32_t sfp = (peer_sfp & MASK) | ((local_sfp & MASK) >> 8);
        tOrSFP_uint32_view(_0{}, mma_m) = sfp;
      }
    }
  }

  template <typename TiledMma, typename TensorRP, typename TensorRSFP, typename TensorRVt,
            typename TensorRSFVt, typename TensorRO, typename TensorAccConv, typename TensorMaxP,
            typename SmemTiledCopyV, typename SmemTiledCopySFV, typename TensorSsVt,
            typename TensorSsSFVt, typename TensorRVtView, typename TensorRSFVtView,
            typename PipelineStateV>
  __device__ __forceinline__ static void compute_pv_gemm(
      TiledMma const& tiled_mma_pv, TensorRP& tOrP, TensorRSFP& tOrSFP, TensorRVt const& tOrVt,
      TensorRSFVt const& tOrSFVt, TensorRO& tOrO, TensorAccConv const& acc_conversion_view,
      TensorMaxP const& AbsMaxP, SmemTiledCopyV const& smem_tiled_copy_V,
      SmemTiledCopySFV const& smem_tiled_copy_SFV, TensorSsVt const& tOsVt,
      TensorSsSFVt const& tOsSFVt, TensorRVtView& tOrVt_copy_view,
      TensorRSFVtView& tOrSFVt_copy_view, PipelineStateV const& smem_pipe_read_v) {
    CUTLASS_PRAGMA_UNROLL
    for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
      cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                 make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)), tOrO);

      if (v_block < size<2>(tOrP) - 1) {
        copy_v_block(smem_tiled_copy_V, smem_tiled_copy_SFV, tOsVt, tOsSFVt, tOrVt_copy_view,
                     tOrSFVt_copy_view, smem_pipe_read_v, v_block + 1);

        quantize_p(acc_conversion_view, AbsMaxP, tOrP, tOrSFP, v_block + 1);
      }
    }
  }

  template <typename TiledMma, typename TensorRP, typename TensorRSFP, typename TensorRVt,
            typename TensorRSFVt, typename TensorRO, typename TensorAccConv, typename TensorMaxP,
            typename SmemTiledCopyV, typename SmemTiledCopySFV, typename TensorSsVt,
            typename TensorSsSFVt, typename TensorRVtView, typename TensorRSFVtView,
            typename PipelineV, typename PipelineStateV>
  __device__ __forceinline__ static void run(
      TiledMma const& tiled_mma_pv, TensorRP& tOrP, TensorRSFP& tOrSFP, TensorRVt const& tOrVt,
      TensorRSFVt const& tOrSFVt, TensorRO& tOrO, TensorAccConv const& acc_conversion_view,
      TensorMaxP const& AbsMaxP, SmemTiledCopyV const& smem_tiled_copy_V,
      SmemTiledCopySFV const& smem_tiled_copy_SFV, TensorSsVt const& tOsVt,
      TensorSsSFVt const& tOsSFVt, TensorRVtView& tOrVt_copy_view,
      TensorRSFVtView& tOrSFVt_copy_view, PipelineV& pipeline_v, PipelineStateV& smem_pipe_read_v) {
    auto barrier_token = pipeline_v.consumer_try_wait(smem_pipe_read_v);
    pipeline_v.consumer_wait(smem_pipe_read_v, barrier_token);

    copy_v_block(smem_tiled_copy_V, smem_tiled_copy_SFV, tOsVt, tOsSFVt, tOrVt_copy_view,
                 tOrSFVt_copy_view, smem_pipe_read_v, _0{});

    quantize_p(acc_conversion_view, AbsMaxP, tOrP, tOrSFP, 0);

    compute_pv_gemm(tiled_mma_pv, tOrP, tOrSFP, tOrVt, tOrSFVt, tOrO, acc_conversion_view, AbsMaxP,
                    smem_tiled_copy_V, smem_tiled_copy_SFV, tOsVt, tOsSFVt, tOrVt_copy_view,
                    tOrSFVt_copy_view, smem_pipe_read_v);

    pipeline_v.consumer_release(smem_pipe_read_v);
    ++smem_pipe_read_v;
  }
};

}  // namespace nvfp4_attention
