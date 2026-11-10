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

#include "../../utils/layout.cuh"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace nvfp4_attention {

using cute::_;
using cute::_0;
using cute::copy;
using cute::get;
using cute::make_identity_tensor;
using cute::make_zip_tensor;
using cute::select;
using cute::size;
using cute::Tensor;

template <typename Traits, bool IsCausal>
struct QKGemmComputer {
  using Element = typename Traits::Element;
  using ElementSF = typename Traits::ElementSF;
  using TileShape_MNK = typename Traits::TileShape_MNK;
  using TiledMmaQK = typename Traits::TiledMmaQK;
  using SmemCopyAtomQ = typename Traits::SmemCopyAtomQ;
  using SmemCopyAtomKV = typename Traits::SmemCopyAtomKV;
  using SmemCopyAtomSF = typename Traits::SmemCopyAtomSF;

  static constexpr int kBlockM = get<0>(TileShape_MNK{});
  static constexpr int kBlockN = get<1>(TileShape_MNK{});
  static constexpr int kBlockK = get<2>(TileShape_MNK{});

  template <typename SmemTiledCopyK, typename SmemTiledCopySFK, typename TensorSsK,
            typename TensorSsSFK, typename TensorSrK, typename TensorSrSFK, typename PipelineStateK>
  __device__ __forceinline__ static void copy_k_block(
      SmemTiledCopyK const& smem_tiled_copy_K, SmemTiledCopySFK const& smem_tiled_copy_SFK,
      TensorSsK const& tSsK, TensorSsSFK const& tSsSFK, TensorSrK& tSrK_copy_view,
      TensorSrSFK& tSrSFK_copy_view, PipelineStateK const& smem_pipe_read_k, auto block_id) {
    auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());
    auto tSsSFK_stage = tSsSFK(_, _, _, smem_pipe_read_k.index());

    copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
    copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
  }

  template <typename TiledMma, typename TensorRQ, typename TensorRSFQ, typename TensorRK,
            typename TensorRSFK, typename TensorRS, typename SmemTiledCopyK,
            typename SmemTiledCopySFK, typename TensorSsK, typename TensorSsSFK,
            typename TensorRKView, typename TensorRSFKView, typename PipelineStateK>
  __device__ __forceinline__ static void compute_qk_gemm(
      TiledMma const& tiled_mma_qk, TensorRQ const& tSrQ, TensorRSFQ const& tSrSFQ,
      TensorRK const& tSrK, TensorRSFK const& tSrSFK, TensorRS& tSrS,
      SmemTiledCopyK const& smem_tiled_copy_K, SmemTiledCopySFK const& smem_tiled_copy_SFK,
      TensorSsK const& tSsK, TensorSsSFK const& tSsSFK, TensorRKView& tSrK_copy_view,
      TensorRSFKView& tSrSFK_copy_view, PipelineStateK const& smem_pipe_read_k) {
    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
      cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                 make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)), tSrS);

      if (k_block < size<2>(tSrQ) - 1) {
        copy_k_block(smem_tiled_copy_K, smem_tiled_copy_SFK, tSsK, tSsSFK, tSrK_copy_view,
                     tSrSFK_copy_view, smem_pipe_read_k, k_block + 1);
      }
    }
  }

  template <typename TensorRS, typename TensorCS>
  __device__ __forceinline__ static void apply_masking(TensorRS& tSrS, TensorCS const& tScS,
                                                       int n_block, int seqlen_k,
                                                       int unpadded_seqlen_k, int seqlen_q,
                                                       int m_block) {
    auto col_limit_causal = [&](int row, int n_block_idx) {
      return row + 1 + seqlen_k - n_block_idx * kBlockN - seqlen_q + m_block * kBlockM;
    };

    int const valid_cols = int(unpadded_seqlen_k - n_block * kBlockN);
    if constexpr (!IsCausal) {
      if (valid_cols >= kBlockN) {
        return;
      }
    } else {
      if (valid_cols >= kBlockN && col_limit_causal(0, n_block) >= kBlockN) {
        return;
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tSrS); ++i) {
      int const col = nvfp4_attention::qk_acc_col_to_k_col(int(get<1>(tScS(i))));
      if constexpr (!IsCausal) {
        if (col >= int(unpadded_seqlen_k - n_block * kBlockN)) {
          tSrS(i) = -INFINITY;
        }
      } else {
        int col_limit =
            std::min(seqlen_k - n_block * kBlockN, col_limit_causal(int(get<0>(tScS(i))), n_block));
        if (col >= col_limit) {
          tSrS(i) = -INFINITY;
        }
      }
    }
  }

  template <typename TiledMma, typename TensorRQ, typename TensorRSFQ, typename TensorRK,
            typename TensorRSFK, typename TensorRS, typename SmemTiledCopyK,
            typename SmemTiledCopySFK, typename TensorSsK, typename TensorSsSFK,
            typename TensorRKView, typename TensorRSFKView, typename PipelineK,
            typename PipelineStateK, typename DeltaSFunc>
  __device__ __forceinline__ static void run(
      TiledMma const& tiled_mma_qk, TensorRQ const& tSrQ, TensorRSFQ const& tSrSFQ,
      TensorRK const& tSrK, TensorRSFK const& tSrSFK, TensorRS& tSrS,
      SmemTiledCopyK const& smem_tiled_copy_K, SmemTiledCopySFK const& smem_tiled_copy_SFK,
      TensorSsK const& tSsK, TensorSsSFK const& tSsSFK, TensorRKView& tSrK_copy_view,
      TensorRSFKView& tSrSFK_copy_view, PipelineK& pipeline_k, PipelineStateK& smem_pipe_read_k,
      int n_block, int seqlen_k, int unpadded_seqlen_k, int seqlen_q, int m_block,
      DeltaSFunc const& add_delta_s_func) {
    auto barrier_token = pipeline_k.consumer_try_wait(smem_pipe_read_k);
    pipeline_k.consumer_wait(smem_pipe_read_k, barrier_token);

    copy_k_block(smem_tiled_copy_K, smem_tiled_copy_SFK, tSsK, tSsSFK, tSrK_copy_view,
                 tSrSFK_copy_view, smem_pipe_read_k, _0{});

    add_delta_s_func(tSrS);

    compute_qk_gemm(tiled_mma_qk, tSrQ, tSrSFQ, tSrK, tSrSFK, tSrS, smem_tiled_copy_K,
                    smem_tiled_copy_SFK, tSsK, tSsSFK, tSrK_copy_view, tSrSFK_copy_view,
                    smem_pipe_read_k);

    Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
    Tensor tScS = tiled_mma_qk.get_thread_slice(threadIdx.x).partition_C(cS);
    apply_masking(tSrS, tScS, n_block, seqlen_k, unpadded_seqlen_k, seqlen_q, m_block);

    pipeline_k.consumer_release(smem_pipe_read_k);
    ++smem_pipe_read_k;
  }
};

}  // namespace nvfp4_attention
