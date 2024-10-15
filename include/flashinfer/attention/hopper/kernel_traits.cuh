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

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVO {
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
  union {
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    int tile_count_semaphore;
  };
};

template <int kStages, class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOVt {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
    union {
      cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v_out;
      cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    typename cutlass::PipelineAsync<kStages>::SharedStorage pipeline_vt;
    int tile_count_semaphore;
    float softmax_scale_qk_log2;
    float descale_v;
  };
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, int kStages_,
          bool Is_Q_in_regs_ = false, int kClusterM_ = 1, typename elem_type = cutlass::half_t>
struct Flash_fwd_kernel_traits {
  using Element = elem_type;
  using ElementAccum = float;
  using OutputType = elem_type;
  using index_t = int64_t;

  // The number of threads.
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarp;

  static constexpr bool Is_Q_in_regs = Is_Q_in_regs_;
  static_assert(kNWarps_ == 4 || kNWarps_ == 8 || kNWarps_ == 12 || kNWarps_ == 16);
  static constexpr bool Is_WS = kNWarps_ >= 12;
  static_assert(!(Is_WS && Is_Q_in_regs), "Warp-specialization does not support Q in registers");

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;

  static constexpr int kClusterM = kClusterM_;
  using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

  static constexpr int kStages = kStages_;

  using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
  using TiledMma0 = decltype(cute::make_tiled_mma(
      std::conditional_t<
          Is_Q_in_regs,
          decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>()),
          decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>())>{},
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
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, Element, decltype(cute::get<1>(TileShape_MNK{})),
                                   decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), Int<kStages>{})));

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutVt = decltype(composition(
      SmemLayoutV{}, make_ordered_layout(make_shape(get<2>(TileShape_MNK{}),
                                                    get<1>(TileShape_MNK{}), Int<kStages>{}),
                                         Step<_2, _1, _3>{})));

  using SmemLayoutAtomO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, OutputType, decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

  using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;

  using SharedStorage = SharedStorageQKVO<kStages, Element, Element, Element, SmemLayoutQ,
                                          SmemLayoutK, SmemLayoutV, SmemLayoutO>;

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<kStages>;
  using PipelineState = typename cutlass::PipelineState<kStages>;
  // using BarrierType = typename MainloopPipeline::ProducerBarrierType;
};

////////////////////////////////////////////////////////////////////////////////////////////////////