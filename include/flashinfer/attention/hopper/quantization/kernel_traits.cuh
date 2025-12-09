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
    FA3-style FP8 transpose: Same-shape transpose with MN-major TMA load and K-major MMA
    Reference:
   https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp

    Key insight: TMA loads V with transposed gmem strides into MN-major smem layout.
    Then we transpose in-place from MN-major to K-major within the same-shape buffer.
    Both SmemLayoutVtTma and SmemLayoutVtMma have shape (HEAD_DIM, CTA_KV, STAGES).

    For sparse path (cp.async loading), we keep the original (CTA_KV, HEAD_DIM, STAGES) layout
    since cp.async loads V directly in its original gmem layout (N, D).
*/
template <typename TileShape_QKD, typename Element, int NUM_STAGES>
struct TranposeTraits_64x64 {
  using TransposeShapeAtom_ = Shape<_64, _64>;
  using TransElement = Element;
  static_assert(cutlass::sizeof_bits_v<TransElement> == 8);

  static constexpr int kHeadDim = get<2>(TileShape_QKD{});
  static constexpr int kBlockN = get<1>(TileShape_QKD{});

  // MN-major for TMA loading (V is loaded with transposed gmem strides)
  static constexpr cute::GMMA::Major TmaMajorV = GMMA::Major::MN;
  // K-major for MMA consumption (required for FP8)
  static constexpr cute::GMMA::Major MmaMajorV = GMMA::Major::K;

  // ==================== TMA Path Layouts (FA3-style same-shape) ====================
  // SmemLayoutVtTma: MN-major layout for TMA load, shape (HEAD_DIM, CTA_KV, STAGES)
  using SmemLayoutAtomVtTma =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<TmaMajorV, Element,
                                                                   Int<kHeadDim>, Int<kBlockN>>());
  using SmemLayoutVtTma = decltype(tile_to_shape(
      SmemLayoutAtomVtTma{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, Int<NUM_STAGES>{}),
      cute::Step<_2, _1, _3>{}));  // MN-major ordering

  // SmemLayoutVtMma: K-major layout for MMA, same shape (HEAD_DIM, CTA_KV, STAGES)
  using SmemLayoutAtomVtMma =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<MmaMajorV, Element,
                                                                   Int<kHeadDim>, Int<kBlockN>>());
  using SmemLayoutVtMma = decltype(tile_to_shape(
      SmemLayoutAtomVtMma{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, Int<NUM_STAGES>{}),
      cute::Step<_1, _2, _3>{}));  // K-major ordering

  // For TMA path: SmemLayoutV = SmemLayoutVtTma (MN-major, for TMA load)
  using SmemLayoutV = SmemLayoutVtTma;
  using SmemLayoutVt = SmemLayoutVtMma;

  // FA3-style LDSM/STSM tiled copies for TMA path transpose
  static constexpr bool kHeadDimMultiple64 = kHeadDim % 64 == 0;
  static_assert(kHeadDimMultiple64 || kBlockN % 64 == 0,
                "Either kHeadDim or kBlockN must be multiple of 64");

  using LDSM_thread_shape =
      std::conditional_t<kHeadDimMultiple64, Shape<_32, _4, _1, _1>, Shape<_16, _4, _1, _2>>;
  using LDSM_thread_stride =
      std::conditional_t<kHeadDimMultiple64, Stride<_4, _1, _0, _0>, Stride<_4, _1, _0, _64>>;
  using LDSM_value_shape = Shape<_2, _2, _1, _4>;
  using LDSM_value_stride = Stride<_1, _2, _16, _4>;
  using LDSM_divide_shape = std::conditional_t<kHeadDimMultiple64, Shape<_64, _8>, Shape<_32, _8>>;

  using S2RTiledCopyVt = decltype(make_tiled_copy(Copy_Atom<SM75_U16x8_LDSM_T, Element>{},
                                                  Layout<LDSM_thread_shape, LDSM_thread_stride>{},
                                                  Layout<LDSM_value_shape, LDSM_value_stride>{}));

  using STSM_thread_shape =
      std::conditional_t<kHeadDimMultiple64, Shape<_8, _4, _4, _1>, Shape<_8, _4, _2, _2>>;
  using STSM_thread_stride =
      std::conditional_t<kHeadDimMultiple64, Stride<_4, _1, _32, _0>, Stride<_4, _1, _32, _64>>;
  using STSM_value_shape = Shape<_1, _4, _2, _2>;
  using STSM_value_stride = Stride<_0, _1, _4, _8>;
  using STSM_divide_shape = Shape<_8, _16>;

  using R2STiledCopyV = decltype(make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, Element>{},
                                                 Layout<STSM_thread_shape, STSM_thread_stride>{},
                                                 Layout<STSM_value_shape, STSM_value_stride>{}));

  // TMA path transpose layouts
  using SmemLayoutVTransposeSrc = SmemLayoutVtTma;
  using SmemLayoutVtTransposeTgt = SmemLayoutVtMma;

  // ==================== Sparse Path Layouts (Original different-shape) ====================
  // For sparse path, cp.async loads V in original (N, D) layout, so we need (CTA_KV, HEAD_DIM,
  // STAGES)
  using SmemLayoutAtomVSparse =
      decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtom_{}));
  using SmemLayoutVSparse = decltype(tile_to_shape(
      SmemLayoutAtomVSparse{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<NUM_STAGES>{})));

  // Sparse path transpose source layout (from SmemLayoutVSparse)
  using SmemShapeLDSM = Shape<Shape<_8, _8>, Shape<_16, _4>>;
  using SmemLayoutDivideVSparse =
      decltype(tiled_divide(SmemLayoutVSparse{}, TransposeShapeAtom_{}));
  using FactoringShapeVSparse = decltype(make_shape(
      SmemShapeLDSM{}, shape<1>(SmemLayoutDivideVSparse{}), shape<2>(SmemLayoutDivideVSparse{}),
      shape<3>(SmemLayoutDivideVSparse{})));
  using SmemLayoutVSparseTransposeSrc =
      decltype(composition(SmemLayoutDivideVSparse{}, make_layout(FactoringShapeVSparse{})));

  // Sparse path transpose target layout (same SmemLayoutVt as TMA path for MMA)
  using SmemLayoutAtomVtSparse =
      decltype(tile_to_shape(GMMA::Layout_K_SW64_Atom<Element>{}, TransposeShapeAtom_{}));
  using SmemLayoutVtSparse = decltype(tile_to_shape(
      SmemLayoutAtomVtSparse{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}, Int<NUM_STAGES>{})));
  using SmemLayoutVtSparseTrans = decltype(composition(
      SmemLayoutVtSparse{},
      make_ordered_layout(product_each(shape(SmemLayoutVSparse{})), Step<_2, _1, _3>{})));
  using SmemLayoutDivideVtSparse =
      decltype(tiled_divide(SmemLayoutVtSparseTrans{}, TransposeShapeAtom_{}));
  using SmemShapeSTSM = Shape<Shape<_16, _4>, Shape<_16, _4>>;
  using FactoringShapeVtSparse = decltype(make_shape(
      SmemShapeSTSM{}, shape<1>(SmemLayoutDivideVtSparse{}), shape<2>(SmemLayoutDivideVtSparse{}),
      shape<3>(SmemLayoutDivideVtSparse{})));
  using SmemLayoutVtSparseTransposeTgt =
      decltype(composition(SmemLayoutDivideVtSparse{}, make_layout(FactoringShapeVtSparse{})));
};

/*
  FA3-style in-kernel transpose of smemV (MN-major) into smemVt (K-major) using LDSM.T & STSM.
  Both tensors have the same shape (HEAD_DIM, CTA_KV, STAGES), only different swizzle patterns.
  Reference:
  https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp

  This is used for TMA path where V is loaded with transposed gmem strides.
*/
template <typename Ktraits>
struct SmemTransposeFP8_64x64 {
  using Element = typename Ktraits::DTypeKV;
  using VTranposeTraits = typename Ktraits::VTranposeTraits;
  using SmemLayoutVtTma = typename Ktraits::SmemLayoutV;
  using SmemLayoutVtMma = typename Ktraits::SmemLayoutVt;
  static_assert(cutlass::sizeof_bits_v<Element> == 8);

  using S2RTiledCopyVt = typename VTranposeTraits::S2RTiledCopyVt;
  using R2STiledCopyV = typename VTranposeTraits::R2STiledCopyV;
  using LDSM_divide_shape = typename VTranposeTraits::LDSM_divide_shape;
  using STSM_divide_shape = typename VTranposeTraits::STSM_divide_shape;

  S2RTiledCopyVt s2r_tiled_copy_vt;
  R2STiledCopyV r2s_tiled_copy_v;

  template <class SmemTensorVt, class SmemTensorV>
  CUTLASS_DEVICE void do_transpose(SmemTensorVt& sVt, SmemTensorV& sV, int stage_idx) {
    using namespace cute;

    auto s2r_thr_copy_vt = s2r_tiled_copy_vt.get_thread_slice(threadIdx.x);
    auto r2s_thr_copy_v = r2s_tiled_copy_v.get_thread_slice(threadIdx.x);

    // flat_divide sVt (source, MN-major) and sV (target, K-major) for transpose
    // sVt shape: (HEAD_DIM, CTA_KV, STAGES)
    // After flat_divide: (LDSM_divide_shape, HEAD_DIM / LDSM_divide_shape[0], CTA_KV /
    // LDSM_divide_shape[1], STAGES)
    Tensor tTranssVt_ = s2r_thr_copy_vt.partition_S(flat_divide(sVt, LDSM_divide_shape{}));
    Tensor tTranssV_ = r2s_thr_copy_v.partition_D(flat_divide(sV, STSM_divide_shape{}));

    // Use ILP=2 for better instruction-level parallelism
    static constexpr int Transpose_ILP =
        (size<2>(tTranssVt_) * size<3>(tTranssVt_)) % 2 == 0 ? 2 : 1;
    Tensor tTranssVt = logical_divide(group_modes<1, rank(tTranssVt_) - 1>(tTranssVt_),
                                      Shape<Underscore, Int<Transpose_ILP>>{});
    Tensor tTranssV = logical_divide(group_modes<1, rank(tTranssV_) - 1>(tTranssV_),
                                     Shape<Underscore, Int<Transpose_ILP>>{});

#pragma unroll
    for (int i = 0; i < size<1, 1>(tTranssVt); ++i) {
      Tensor tTransrV = make_fragment_like(tTranssV(_, make_coord(_, _0{}), _0{}));
      static_assert(size<0>(tTransrV) == 16);
      Tensor tTransrV_64 = recast<uint2>(tTransrV);

      // Load from MN-major smem using LDSM.T
      cute::copy(s2r_tiled_copy_vt, tTranssVt(_, make_coord(_, i), stage_idx), tTransrV);

// Byte permutation for FP8 element reordering
#pragma unroll
      for (int j = 0; j < size(tTransrV_64); ++j) {
        uint32_t upper = tTransrV_64[j].x;
        uint32_t lower = tTransrV_64[j].y;
        tTransrV_64[j].x = __byte_perm(upper, lower, 0x6420);
        tTransrV_64[j].y = __byte_perm(upper, lower, 0x7531);
      }

      // Store to K-major smem using STSM
      cute::copy(r2s_tiled_copy_v, tTransrV, tTranssV(_, make_coord(_, i), stage_idx));
    }

    // Sync all WG threads for ldmatrix completion
    cutlass::arch::NamedBarrier::sync(Ktraits::NUM_PRODUCER_THREADS,
                                      static_cast<int>(NamedBarriers::kProducerWG) /*id*/);
  }

  // Legacy interface for backward compatibility
  using SmemLayoutVTransposeSrc = SmemLayoutVtTma;
  using SmemLayoutVtTransposeTgt = SmemLayoutVtMma;
};

/*
  Original FP8 transpose for sparse path (cp.async loading).
  V is loaded in original (N, D) layout via cp.async, so smemV has shape (CTA_KV, HEAD_DIM, STAGES).
  Transpose to smemVt with shape (HEAD_DIM, CTA_KV, STAGES) for MMA consumption.
*/
template <typename Ktraits>
struct SmemTransposeFP8_64x64_Sparse {
  using Element = typename Ktraits::DTypeKV;
  using SmemLayoutVSparseTransposeSrc = typename Ktraits::SmemLayoutVSparseTransposeSrc;
  using SmemLayoutVtSparseTransposeTgt = typename Ktraits::SmemLayoutVtSparseTransposeTgt;
  static_assert(cutlass::sizeof_bits_v<Element> == 8);

  using ldsm_thread_shape = Shape<_4, _1, _8, _4>;
  using ldsm_value_shape = Shape<_2, _8, _2, _1>;
  using ldsm_value_stride = Stride<_2, _4, _1, _0>;
  // use trans to do 16bits transpose
  // which needs permutation to separate 8bits row and column
  using TiledCopyLDSM =
      decltype(make_tiled_copy(Copy_Atom<SM75_U16x8_LDSM_T, Element>{}, Layout<ldsm_thread_shape>{},
                               Layout<ldsm_value_shape, ldsm_value_stride>{}));
  TiledCopyLDSM tiled_copy_ldsm;

  using stsm_thread_shape = Shape<_4, _1, _8, _4>;
  using stsm_value_shape = Shape<_4, _4, _2, _1>;
  using stsm_value_stride = Stride<_1, _8, _4, _0>;

  using TiledCopySTSM =
      decltype(make_tiled_copy(Copy_Atom<SM90_U32x4_STSM_N, Element>{}, Layout<stsm_thread_shape>{},
                               Layout<stsm_value_shape, stsm_value_stride>{}));
  TiledCopySTSM tiled_copy_stsm;

  template <class SmemTensor, class SmemTensorOut>
  CUTLASS_DEVICE void _tranpose(SmemTensor&& s_in, SmemTensorOut&& s_out) {
    using namespace cute;

    auto tid = threadIdx.x;
    auto thr_copy_ldsm = tiled_copy_ldsm.get_thread_slice(tid);
    auto thr_copy_stsm = tiled_copy_stsm.get_thread_slice(tid);

    auto tXsX = thr_copy_ldsm.partition_S(s_in);
    auto tXrX = make_tensor<Element>(shape(tXsX));
    auto tXsX_out = thr_copy_stsm.partition_D(s_out);

    cute::copy(tiled_copy_ldsm, tXsX, tXrX);
    auto data = tXrX.data();
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size(tXrX); n += 8) {
      uint32_t* data_32bit = reinterpret_cast<uint32_t*>(&data[n]);
      auto upper = data_32bit[0];
      auto lower = data_32bit[1];
      // select row-major elements.
      // from (0 1 16 17) (128 129 144 145) to (0 16 128 144) (1 17 129 145)
      // which is (0 1 8 9)
      data_32bit[0] = __byte_perm(upper, lower, 0x6420);
      data_32bit[1] = __byte_perm(upper, lower, 0x7531);
    }
    cute::copy(tiled_copy_stsm, tXrX, tXsX_out);
  }

  template <class SmemTensor, class SmemTensorOut>
  CUTLASS_DEVICE void do_transpose(SmemTensor& s_in, SmemTensorOut& s_out, int stage_idx) {
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < shape<2>(SmemLayoutVSparseTransposeSrc{}); ++j) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < shape<1>(SmemLayoutVSparseTransposeSrc{}); ++i) {
        this->_tranpose(flatten(s_in(_, i, j, stage_idx)), flatten(s_out(_, i, j, stage_idx)));
      }
    }
    // For FP8 kernel, all WG threads will arrive for issuing ldmatrix
    cutlass::arch::NamedBarrier::sync(Ktraits::NUM_PRODUCER_THREADS,
                                      static_cast<int>(NamedBarriers::kProducerWG) /*id*/);
  }
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
  // TMA path layouts (FA3-style same-shape transpose)
  using SmemLayoutV = typename VTranposeTraits::SmemLayoutV;
  using SmemLayoutVt = typename VTranposeTraits::SmemLayoutVt;
  using SmemLayoutVTransposeSrc = typename VTranposeTraits::SmemLayoutVTransposeSrc;
  using SmemLayoutVtTransposeTgt = typename VTranposeTraits::SmemLayoutVtTransposeTgt;

  // Sparse path layouts (original different-shape transpose)
  using SmemLayoutVSparse = typename VTranposeTraits::SmemLayoutVSparse;
  using SmemLayoutVtSparse = typename VTranposeTraits::SmemLayoutVtSparse;
  using SmemLayoutVSparseTransposeSrc = typename VTranposeTraits::SmemLayoutVSparseTransposeSrc;
  using SmemLayoutVtSparseTransposeTgt = typename VTranposeTraits::SmemLayoutVtSparseTransposeTgt;

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
  // NOTE: Use SmemLayoutVSparse for SharedStorage to ensure sparse (paged KV) path works correctly.
  // SmemLayoutVSparse has shape (CTA_KV, HEAD_DIM, STAGES) which matches cp.async loading pattern.
  // For TMA path, we create the tensor with SmemLayoutV (FA3-style) layout in mainloop_load.cuh.
  // Both layouts have the same cosize, so memory allocation is identical.
  using SharedStorage =
      SharedStorageQKVOVt<MainloopPipeline, DTypeQ, DTypeKV, DTypeO, IdType, CTA_KV, SmemLayoutQ,
                          SmemLayoutK, SmemLayoutVSparse, SmemLayoutO>;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_FP8_KERNEL_TRAITS_CUH_
