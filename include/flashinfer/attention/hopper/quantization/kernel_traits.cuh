/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_FP8_KERNEL_TRAITS_CUH_
#define FLASHINFER_ATTENTION_HOPPER_FP8_KERNEL_TRAITS_CUH_

#include <type_traits>

#include "../../../cutlass_utils.cuh"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace flashinfer {

using namespace cute;

/*
    Add additional smem for Vt
    NOTE(Yilong): Should modify the mainloop to leverage the smem_v_read's early release
*/
template <typename MainloopPipeline, class DTypeQ, class DTypeKV, class DTypeOut, class IdType,
          int CTA_KV, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV, class SmemLayoutO>
struct SharedStorageQKVOVt {
  cute::array_aligned<DTypeQ, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutV>> smem_v;
  union {
    cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutV>> smem_vt;
    cute::array_aligned<DTypeOut, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    cutlass::arch::ClusterBarrier barrier_O;
    typename MainloopPipeline::SharedStorage pipeline_k;
    typename MainloopPipeline::SharedStorage pipeline_v;
    // vt only use ldmatrix, which do not need TMA Pipeline
    typename cutlass::PipelineAsync<MainloopPipeline::Stages>::SharedStorage pipeline_vt;
  };
};

/*
    In-kernel FP8 transpose adopted from FlashAttention-3 template
   https://github.com/Dao-AILab/flash-attention/blob/c7f32a8409e52a84bd8046afe7060da33036f9a5/hopper/kernel_traits.h#L217
*/
template <typename TileShape_QKD, typename Element, int NUM_STAGES>
struct TranposeTraits_64x64 {
  using TransposeShapeAtom_ = Shape<_64, _64>;
  using TransElement = Element;
  static_assert(cutlass::sizeof_bits_v<TransElement> == 8);

  using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
  using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;

  using SmemLayoutAtomV =
      decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<TransElement>{}, TransposeShapeAtom_{}));
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(get<1>(TileShape_QKD{}), get<2>(TileShape_QKD{}), Int<NUM_STAGES>{})));
  using SmemLayoutDivideV = decltype(tiled_divide(SmemLayoutV{}, TransposeShapeAtom_{}));
  using FactoringShapeV =
      decltype(make_shape(SmemShapeLDSM{}, shape<1>(SmemLayoutDivideV{}),
                          shape<2>(SmemLayoutDivideV{}), shape<3>(SmemLayoutDivideV{})));
  using SmemLayoutVTransposeSrc =
      decltype(composition(SmemLayoutDivideV{}, make_layout(FactoringShapeV{})));

  using SmemLayoutAtomVt =
      decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtom_{}));
  // k-major Vt as target layout. this changes the memory
  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(get<2>(TileShape_QKD{}), get<1>(TileShape_QKD{}), Int<NUM_STAGES>{})));
  using SmemLayoutVtTrans = decltype(composition(
      SmemLayoutVt{}, make_ordered_layout(product_each(shape(SmemLayoutV{})), Step<_2, _1, _3>{})));
  using SmemLayoutDivideVt = decltype(tiled_divide(SmemLayoutVtTrans{}, TransposeShapeAtom_{}));
  using FactoringShapeVt =
      decltype(make_shape(SmemShapeSTSM{}, shape<1>(SmemLayoutDivideVt{}),
                          shape<2>(SmemLayoutDivideVt{}), shape<3>(SmemLayoutDivideVt{})));
  using SmemLayoutVtTransposeTgt =
      decltype(composition(SmemLayoutDivideVt{}, make_layout(FactoringShapeVt{})));
};

template <bool USE_TMA_LOAD_KV, int HEAD_DIM_, int CTA_Q_, int CTA_KV_, int NUM_STAGES_,
          typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_,
          typename AttentionVariant_>
struct FP8AttentionKernelTraits {
  using AttentionVariant = AttentionVariant_;

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;

  static constexpr int CTA_Q = CTA_Q_;
  static_assert(CTA_Q % 64 == 0);
  static constexpr int CTA_KV = CTA_KV_;
  static constexpr int HEAD_DIM = HEAD_DIM_;
  static_assert(HEAD_DIM % 32 == 0);

  static constexpr int NUM_WARPS = ((CTA_Q / 64) + 1) * 4;
  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;
  // NOTE(Zihao): the following constant should only be used when TMA is enabled,
  // where only one warp inside a warp group is used for TMA.

  // In FP16 kernel, only one thread of single warp within the producer WG is working
  // For FP8, we use the entire WG for tranposing V
  static constexpr int NUM_PRODUCER_THREADS = cutlass::NumThreadsPerWarpGroup;

  using TileShape_QKD = Shape<Int<CTA_Q>, Int<CTA_KV>, Int<HEAD_DIM>>;

  static constexpr int NUM_STAGES = NUM_STAGES_;

  using AtomLayoutQKD = Layout<Shape<Int<CTA_Q / 64>, _1, _1>>;
  using TiledMmaQK = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<DTypeQ, DTypeKV, DTypeQKAccum, TileShape_QKD>(), AtomLayoutQKD{}));

  // FP8 needs K-major for both P / V
  using TiledMmaPV = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<DTypeKV, DTypeKV, /*ElementAccum=*/float,
                                 decltype(select<0, 2, 1>(TileShape_QKD{})), GMMA::Major::K,
                                 GMMA::Major::K>(),
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

  using VTranposeTraits = TranposeTraits_64x64<TileShape_QKD, DTypeKV, NUM_STAGES>;
  using SmemLayoutV = typename VTranposeTraits::SmemLayoutV;
  using SmemLayoutVt = typename VTranposeTraits::SmemLayoutVt;
  using SmemLayoutVTransposeSrc = typename VTranposeTraits::SmemLayoutVTransposeSrc;
  using SmemLayoutVtTransposeTgt = typename VTranposeTraits::SmemLayoutVtTransposeTgt;

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeO, decltype(cute::get<0>(TileShape_QKD{})),
                                   decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_QKD{})));
  using MainloopPipeline =
      std::conditional_t<USE_TMA_LOAD_KV, typename cutlass::PipelineTmaAsync<NUM_STAGES>,
                         typename cutlass::PipelineAsync<NUM_STAGES>>;
  using MainloopPipelineNoTMA = typename cutlass::PipelineAsync<NUM_STAGES>;
  using PipelineState = typename cutlass::PipelineState<NUM_STAGES>;

  // Modify SharedStorage
  using SharedStorage =
      SharedStorageQKVOVt<MainloopPipeline, DTypeQ, DTypeKV, DTypeO, IdType, CTA_KV, SmemLayoutQ,
                          SmemLayoutK, SmemLayoutV, SmemLayoutO>;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_FP8_KERNEL_TRAITS_CUH_
