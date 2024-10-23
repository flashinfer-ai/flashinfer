/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 *Dao.
 ******************************************************************************/

#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

template <int NUM_STAGES, class Gemm1Type, class Gemm2Type, class DTypeOut, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVO {
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
  union {
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<DTypeOut, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<NUM_STAGES>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<NUM_STAGES>::SharedStorage pipeline_v;
  };
};

template <int HEAD_DIM_, int CTA_Q_, int CTA_KV_, int NUM_WARPS_, int NUM_STAGES_,
          typename DType_ = cutlass::half_t>
struct AttentionKernelTraits {
  using Element = DType_;
  using ElementAccum = float;
  using OutputType = DType_;
  using index_t = int64_t;

  // The number of threads.
  static constexpr int kNWarps = NUM_WARPS_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;

  static_assert(NUM_WARPS_ == 4 || NUM_WARPS_ == 8 || NUM_WARPS_ == 12 || NUM_WARPS_ == 16);
  static constexpr bool Is_WS = NUM_WARPS_ >= 12;

  static constexpr int CTA_Q = CTA_Q_;
  static constexpr int CTA_KV = CTA_KV_;
  static constexpr int HEAD_DIM = HEAD_DIM_;
  static_assert(HEAD_DIM % 32 == 0);
  using TileShape_MNK = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEAD_DIM>>;

  using ClusterShape_MNK = Shape<_1, _1, _1>;

  static constexpr int NUM_STAGES = NUM_STAGES_;

  using AtomLayoutMNK = Layout<Shape<Int<CTA_Q / 64>, _1, _1>>;
  using TiledMma0 = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
      AtomLayoutMNK{}));
  using TiledMma1 = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<Element, Element, ElementAccum,
                                 decltype(select<0, 2, 1>(TileShape_MNK{})), GMMA::Major::K,
                                 GMMA::Major::MN>(),
      AtomLayoutMNK{}));

  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, Element, decltype(cute::get<0>(TileShape_MNK{})),
                                   decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, Element, decltype(cute::get<1>(TileShape_MNK{})),
                                   decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<NUM_STAGES>{})));

  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, Element, decltype(cute::get<1>(TileShape_MNK{})),
                                   decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), Int<NUM_STAGES>{})));

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{}, make_ordered_layout(make_shape(get<2>(TileShape_MNK{}),
                                                    get<1>(TileShape_MNK{}), Int<NUM_STAGES>{}),
                                         Step<_2, _1, _3>{})));

  using SmemLayoutAtomO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, OutputType, decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

  using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

  using SharedStorage = SharedStorageQKVO<NUM_STAGES, Element, Element, Element, SmemLayoutQ,
                                          SmemLayoutK, SmemLayoutV, SmemLayoutO>;

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<NUM_STAGES>;
  using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<NUM_STAGES>;
  using PipelineState = typename cutlass::PipelineState<NUM_STAGES>;
};

////////////////////////////////////////////////////////////////////////////////////////////////////