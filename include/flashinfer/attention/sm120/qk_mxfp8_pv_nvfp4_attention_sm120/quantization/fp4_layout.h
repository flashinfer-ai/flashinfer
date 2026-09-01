/*
 * Copyright (c) 2025 by SageAttention team.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/int_tuple.hpp"
#include "cutlass/layout/matrix.h"

namespace qk_mxfp8_pv_nvfp4_attention {

using namespace cute;

/**
 * FP4 Block-Scaled Layout Configuration
 *
 *
 * Block Structure:
 *
 *
 */

/**
 *
 *
 * @tparam major - Major order (K-major by default)
 */
template <int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct BlockScaledBasicChunk;

// Specialization for SFVecSize=16 (NVFP4 / mxf4nvf4)
// K_atom=64, 64/16=4 SF per K-atom
// Block: 64 elements in MN, 4 SF per unit
template <UMMA::Major major>
struct BlockScaledBasicChunk<16, major> {
  using Blk_MN = _64;
  using Blk_SF = _4;
  static constexpr int MMA_NSF = 4;

  using mnBasicBlockShape = Shape<_16, _4>;
  using mnBasicBlockStride = Stride<_16, _4>;
  using kBasicBlockShape = Shape<_16, _4>;  // (SFVecSize, MMA_NSF)
  using kBasicBlockStride = Stride<_0, _1>;

  // Shape: ((16, 4), (16, 4))   Stride: ((16, 4), (0, 1))
  using SfAtom = Layout<Shape<mnBasicBlockShape, kBasicBlockShape>,
                        Stride<mnBasicBlockStride, kBasicBlockStride>>;
};

// Specialization for SFVecSize=32 (FP8 / mxf8f6f4)
// Mirrors NVFP4 BlockScaledBasicChunk<16> structure scaled for SFVecSize=32.
// Base atom: SM120_16x8x32_TN_VS (K=32, N=8), tiled 4× in N by make_tiled_mma.
// kBasicBlock: 32 K-elements per SF group, 4 groups (HeadDim/32=4).
template <UMMA::Major major>
struct BlockScaledBasicChunk<32, major> {
  using Blk_MN = _128;
  using Blk_SF = _4;
  static constexpr int MMA_NSF = 4;  // 4 SF groups per kBasicBlock (matches NVFP4 pattern)

  using mnBasicBlockShape = Shape<_32, _4>;
  using mnBasicBlockStride = Stride<_16, _4>;
  using kBasicBlockShape = Shape<_32, _4>;
  using kBasicBlockStride = Stride<_0, _1>;

  using SfAtom =
      Layout<Shape<Shape<_32, _4>, Shape<_32, _4>>, Stride<Stride<_16, _4>, Stride<_0, _1>>>;
};

/**
 *
 *
 *
 */
template <int SFVecSize_>
struct BlockScaledConfig {
  static constexpr int SFVecSize = SFVecSize_;

  using BlkScaledChunk = BlockScaledBasicChunk<SFVecSize>;

  // Pull all constants from the chunk specialization
  static constexpr int MMA_NSF = BlkScaledChunk::MMA_NSF;

  using Blk_MN = typename BlkScaledChunk::Blk_MN;
  using Blk_SF = typename BlkScaledChunk::Blk_SF;

  using mnBasicBlockShape = typename BlkScaledChunk::mnBasicBlockShape;
  using mnBasicBlockStride = typename BlkScaledChunk::mnBasicBlockStride;
  using kBasicBlockShape = typename BlkScaledChunk::kBasicBlockShape;
  using kBasicBlockStride = typename BlkScaledChunk::kBasicBlockStride;

  using SfAtom = typename BlkScaledChunk::SfAtom;

  // Global memory layout for scale factors
  using LayoutSF = decltype(blocked_product(
      SfAtom{}, make_layout(make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                            make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))));

  // Elements per block
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

  // Shared memory stride for M/N dimensions
  using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));

  /**
   * Tile Scale Factor Atom to Q/K/V Shape
   *
   *
   * @param problem_shape - (Seqlen, Dim, HeadNum, Batch)
   * @return Tiled layout
   */
  template <class ProblemShape>
  CUTE_HOST_DEVICE static constexpr auto tile_atom_to_shape_SFQKV(ProblemShape problem_shape) {
    auto [Seqlen, Dim, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Seqlen, Dim, HeadNum, Batch), Step<_2, _1, _3, _4>{});
  }

  /**
   * Tile Scale Factor Atom to V^T Shape
   *
   *
   * @param problem_shape - (Dim, Seqlen, HeadNum, Batch)
   * @return Tiled layout
   */
  template <class ProblemShape>
  CUTE_HOST_DEVICE static constexpr auto tile_atom_to_shape_SFVt(ProblemShape problem_shape) {
    auto [Dim, Seqlen, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Dim, Seqlen, HeadNum, Batch), Step<_2, _1, _3, _4>{});
  }

  /**
   *
   *
   * @return Shared memory layout
   */
  template <class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFQ(TiledMma tiled_mma,
                                                               TileShape_MNK tileshape_mnk) {
    // K dimension shape
    using sSFQ_shapeK =
        decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                    size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),
                         kBasicBlockShape{}));

    // M dimension shape
    using sSFQ_shapeM = decltype(prepend(size<0>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));

    // Strides
    using sSFQ_strideM = sSF_strideMN;
    using sSFQ_strideK = decltype(prepend(
        make_stride(Int<MMA_NSF>{}, size<0>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}),
        kBasicBlockStride{}));

    // Complete layout
    using sSFQ_shape = decltype(make_shape(sSFQ_shapeM{}, sSFQ_shapeK{}));
    using sSFQ_stride = decltype(make_stride(sSFQ_strideM{}, sSFQ_strideK{}));
    using SmemLayoutAtomSFQ = decltype(make_layout(sSFQ_shape{}, sSFQ_stride{}));

    return SmemLayoutAtomSFQ{};
  }

  /**
   *
   *
   * @return Shared memory layout
   */
  template <class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFKV(TiledMma tiled_mma,
                                                                TileShape_MNK tileshape_mnk) {
    // K dimension shape
    using sSFK_shapeK =
        decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                    size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),
                         kBasicBlockShape{}));

    // N dimension shape
    using sSFK_shapeN = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));

    // Strides
    using sSFK_strideN = sSF_strideMN;
    using sSFK_strideK = decltype(prepend(
        make_stride(Int<MMA_NSF>{}, size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}),
        kBasicBlockStride{}));

    // Complete layout
    using sSFK_shape = decltype(make_shape(sSFK_shapeN{}, sSFK_shapeK{}));
    using sSFK_stride = decltype(make_stride(sSFK_strideN{}, sSFK_strideK{}));
    using SmemLayoutAtomSFK = decltype(make_layout(sSFK_shape{}, sSFK_stride{}));

    return SmemLayoutAtomSFK{};
  }

  /**
   *
   *
   * @return Shared memory layout
   */
  template <class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFVt(TiledMma tiled_mma,
                                                                TileShape_MNK tileshape_mnk) {
    // K dimension shape
    using sSFVt_shapeK =
        decltype(prepend(make_shape(Blk_SF{} / Int<MMA_NSF>{},
                                    size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}),
                         kBasicBlockShape{}));

    // N dimension shape (for V^T)
    using sSFVt_shapeN =
        decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));

    // Strides
    using sSFVt_strideN = sSF_strideMN;
    using sSFVt_strideK = decltype(prepend(
        make_stride(Int<MMA_NSF>{}, size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}),
        kBasicBlockStride{}));

    // Complete layout
    using sSFVt_shape = decltype(make_shape(sSFVt_shapeN{}, sSFVt_shapeK{}));
    using sSFVt_stride = decltype(make_stride(sSFVt_strideN{}, sSFVt_strideK{}));
    using SmemLayoutAtomSFVt = decltype(make_layout(sSFVt_shape{}, sSFVt_stride{}));

    return SmemLayoutAtomSFVt{};
  }
};

}  // namespace qk_mxfp8_pv_nvfp4_attention
