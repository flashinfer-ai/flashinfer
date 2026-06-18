/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_KERNEL_TRAITS_CUH_
#define FLASHINFER_ATTENTION_HOPPER_KERNEL_TRAITS_CUH_

#include <type_traits>

#include "../../cutlass_utils.cuh"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace flashinfer {

using namespace cute;

template <typename MainloopPipeline, class DTypeQ, class DTypeKV, class DTypeOut, class IdType,
          int CTA_KV, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVO {
  cute::array_aligned<DTypeQ, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutK>> smem_k;
  union {
    cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<DTypeOut, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename MainloopPipeline::SharedStorage pipeline_k;
    typename MainloopPipeline::SharedStorage pipeline_v;
  };
};

template <bool USE_TMA_LOAD_KV, int HEAD_DIM_QK_, int HEAD_DIM_VO_, int CTA_Q_, int CTA_KV_,
          int NUM_STAGES_, typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_,
          typename AttentionVariant_>
struct AttentionKernelTraits {
  using AttentionVariant = AttentionVariant_;

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;

  static constexpr int CTA_Q = CTA_Q_;
  static_assert(CTA_Q % 64 == 0);
  static constexpr int CTA_KV = CTA_KV_;
  static constexpr int HEAD_DIM_QK = HEAD_DIM_QK_;
  static constexpr int HEAD_DIM_VO = HEAD_DIM_VO_;
  static_assert(HEAD_DIM_QK % 32 == 0);
  static_assert(HEAD_DIM_VO % 32 == 0);

  static constexpr int NUM_WARPS = ((CTA_Q / 64) + 1) * 4;
  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;
  // NOTE(Zihao): the following constant should only be used when TMA is enabled,
  // where only one warp inside a warp group is used for TMA.
  static constexpr int NUM_PRODUCER_THREADS =
      USE_TMA_LOAD_KV ? cutlass::NumThreadsPerWarp : 4 * cutlass::NumThreadsPerWarp;

  using TileShape_QKD = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEAD_DIM_QK>>;
  using TileShape_PDV = Shape<Int<CTA_Q>, Int<HEAD_DIM_VO>, Int<CTA_KV>>;

  static constexpr int NUM_STAGES = NUM_STAGES_;

  using AtomLayoutQKD = Layout<Shape<Int<CTA_Q / 64>, _1, _1>>;
  using TiledMmaQK = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<DTypeQ, DTypeKV, DTypeQKAccum, TileShape_QKD>(), AtomLayoutQKD{}));
  using TiledMmaPV = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<DTypeKV, DTypeKV, /*ElementAccum=*/float, TileShape_PDV,
                                 GMMA::Major::K, GMMA::Major::MN>(),
      AtomLayoutQKD{}));

  static constexpr int NUM_MMA_THREADS = size(TiledMmaQK{});

  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeQ, decltype(cute::get<0>(TileShape_QKD{})),
                                   decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_QKD{})));

  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeKV, decltype(cute::get<1>(TileShape_QKD{})),
                                   decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(shape<1>(TileShape_QKD{}), shape<2>(TileShape_QKD{}), Int<NUM_STAGES>{})));

  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeKV, decltype(cute::get<2>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(get<2>(TileShape_PDV{}), get<1>(TileShape_PDV{}), Int<NUM_STAGES>{})));

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{}, make_ordered_layout(make_shape(get<1>(TileShape_PDV{}),
                                                    get<2>(TileShape_PDV{}), Int<NUM_STAGES>{}),
                                         Step<_2, _1, _3>{})));

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeO, decltype(cute::get<0>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_PDV{})));
  using MainloopPipeline =
      std::conditional_t<USE_TMA_LOAD_KV, typename cutlass::PipelineTmaAsync<NUM_STAGES>,
                         typename cutlass::PipelineAsync<NUM_STAGES>>;
  using PipelineState = typename cutlass::PipelineState<NUM_STAGES>;

  using SharedStorage = SharedStorageQKVO<MainloopPipeline, DTypeQ, DTypeKV, DTypeO, IdType, CTA_KV,
                                          SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutO>;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_KERNEL_TRAITS_CUH_
