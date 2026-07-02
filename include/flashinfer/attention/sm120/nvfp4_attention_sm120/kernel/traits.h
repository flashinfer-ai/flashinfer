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

#include "../common/cute_extension.h"
#include "../primitives/barrier.cuh"
#include "../quantization/fp4_layout.h"
#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

namespace nvfp4_attention {

template <int kStages, int EpiStages, typename Element, typename ElementSF, typename OutputType,
          typename ElementDS, typename SmemLayoutQ, typename SmemLayoutK, typename SmemLayoutV,
          typename SmemLayoutDS, typename SmemLayoutO, typename SmemLayoutSFQ,
          typename SmemLayoutSFK, typename SmemLayoutSFV>
struct SharedStorageQKVOwithSF : cute::aligned_struct<128, _0> {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> smem_q;

  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> smem_k;

  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_SFQ;
  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_SFK;
  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> smem_SFV;

  alignas(1024) cute::ArrayEngine<ElementDS, cute::cosize_v<SmemLayoutDS>> smem_ds;

  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutV>> smem_v;

  alignas(1024) cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;

  struct {
    alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
    alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    alignas(16)
        typename nvfp4_attention::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>::SharedStorage
        barrier_o;

    alignas(16) typename nvfp4_attention::OrderedSequenceBarrier<2, 2>::SharedStorage math_order;
    int tile_count_semaphore;
  };
};

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kStages_, int kClusterM_, bool BlockMean_,
          typename ElementPairType_ = cutlass::nv_float4_t<cutlass::float_e2m1_t>,
          typename ElementOut_ = cutlass::bfloat16_t, typename ElementDS_ = float>
struct Flash_fwd_kernel_traits {
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr bool BlockMean = BlockMean_;
  static constexpr int kStoreBlockM = kBlockM;
  static constexpr bool SmoothQ = true;

  static_assert(kHeadDim % 32 == 0, "Head dim must be multiple of 32");
  static_assert(kBlockM == 64 || kBlockM == 128, "BlockM must be 64 or 128");

  static constexpr int kNWarps = kBlockM == 128 ? 12 : 8;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

  static constexpr int kBlockMPerWG = kBlockM / 2;

  static constexpr int kClusterM = kClusterM_;
  static constexpr int kStages = kStages_;
  static constexpr int EpiStages = 1;

  static constexpr int NumSFQK = kHeadDim / 16;
  static constexpr int NumSFPV = kBlockN / 16;
  static constexpr auto SFVectorSize = 16;

  using ElementSF = cutlass::float_ue4m3_t;
  using Element = cutlass::float_e2m1_t;
  using ElementAccum = float;
  using ElementOut = ElementOut_;
  using ElementDS = ElementDS_;
  using index_t = int64_t;

  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  using PermTileM = Int<kBlockMPerWG>;
  using PermTileN = _32;
  using PermTileK = Int<kHeadDim>;

  using ElementQMma =
      decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<
               Element>());
  using ElementKMma =
      decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<
               Element>());

  using AtomLayoutMNK = Layout<Shape<_4, _1, _1>>;

  using TiledMmaQK =
      decltype(cute::make_tiled_mma(cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
                                    AtomLayoutMNK{}, Tile<PermTileM, PermTileN, PermTileK>{}));

  using TiledMmaPV =
      decltype(cute::make_tiled_mma(cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
                                    AtomLayoutMNK{}, Tile<PermTileM, _32, PermTileK>{}));

  using AtomLayoutMNK_Full = Layout<Shape<_8, _1, _1>>;
  using TiledMmaQK_Full = decltype(cute::make_tiled_mma(
      cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{}, AtomLayoutMNK_Full{},
      Tile<Int<kBlockM>, PermTileN, PermTileK>{}));
  using TiledMmaPV_Full =
      decltype(cute::make_tiled_mma(cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
                                    AtomLayoutMNK_Full{}, Tile<Int<kBlockM>, _32, PermTileK>{}));
  using AtomLayoutMNK_Store = Layout<Shape<Int<kStoreBlockM / 16>, _1, _1>>;
  using TiledMmaPV_Store = decltype(cute::make_tiled_mma(
      cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{}, AtomLayoutMNK_Store{},
      Tile<Int<kStoreBlockM>, _32, PermTileK>{}));

  static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVectorSize;

  using GmemTiledCopy = SM90_TMA_LOAD;
  using GmemTiledCopySF = SM90_TMA_LOAD;

  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
                                   Element, decltype(size<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
                                   Element, decltype(size<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
                                   Element, decltype(size<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
                                    Element, decltype(size<1>(TileShape_MNK{}))>());

  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

  using SmemLayoutVt = decltype(tile_to_shape(
      SmemLayoutAtomVt{},
      make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

  using SmemLayoutAtomDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<_0, _1>>;
  using SmemLayoutDS = decltype(tile_to_shape(
      SmemLayoutAtomDS{},
      make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

  using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
  using SmemCopyAtomDS = Copy_Atom<UniversalCopy<ElementDS>, ElementDS>;

  using BlkScaledConfig = nvfp4_attention::BlockScaledConfig<SFVectorSize>;
  using LayoutSF = typename BlkScaledConfig::LayoutSF;
  using SfAtom = typename BlkScaledConfig::SfAtom;

  using SmemLayoutAtomSFQ =
      decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));

  using SmemLayoutAtomSFK =
      decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));

  using SmemLayoutAtomSFV =
      decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));

  using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(
      TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));

  using LayoutSFP =
      decltype(make_layout(make_shape(make_shape(_16{}, _4{}), _1{}, Int<kBlockN / 64>{}),
                           make_stride(make_stride(_0{}, _1{}), _0{}, _4{})));

  using LayoutP =
      decltype(make_layout(make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBlockN / 64>{}),
                           make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})));

  using SmemLayoutSFQ =
      decltype(make_layout(shape(SmemLayoutAtomSFQ{}), stride(SmemLayoutAtomSFQ{})));

  using SmemLayoutSFK = decltype(make_layout(
      append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
      append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))));

  using SmemLayoutSFV = decltype(make_layout(
      append(shape(SmemLayoutAtomSFV{}), Int<kStages>{}),
      append(stride(SmemLayoutAtomSFV{}), size(filter_zeros(SmemLayoutAtomSFV{})))));

  using SmemLayoutSFVt = decltype(make_layout(
      append(shape(SmemLayoutAtomSFVt{}), Int<kStages>{}),
      append(stride(SmemLayoutAtomSFVt{}), size(filter_zeros(SmemLayoutAtomSFVt{})))));

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, ElementOut, Int<kStoreBlockM>,
                                   decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO = decltype(tile_to_shape(
      SmemLayoutAtomO{}, make_shape(Int<kStoreBlockM>{}, Int<kHeadDim>{}), Step<_1, _2>{}));

  using SmemLayoutO_Half = decltype(tile_to_shape(
      SmemLayoutAtomO{}, make_shape(Int<kBlockMPerWG>{}, Int<kHeadDim>{}), Step<_1, _2>{}));

  using SharedStorage =
      SharedStorageQKVOwithSF<kStages, EpiStages, Element, ElementSF, ElementOut, ElementDS,
                              SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutDS, SmemLayoutO,
                              SmemLayoutSFQ, SmemLayoutSFK, SmemLayoutSFVt>;

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using PipelineState = typename cutlass::PipelineState<kStages>;

  using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
  using PipelineParamsQ = typename MainloopPipelineQ::Params;
  using PipelineStateQ = typename cutlass::PipelineState<1>;

  using EpilogueBarrier =
      typename nvfp4_attention::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;

  using MathOrderBarrier = nvfp4_attention::OrderedSequenceBarrier<2, 2>;
};

}  // namespace nvfp4_attention
