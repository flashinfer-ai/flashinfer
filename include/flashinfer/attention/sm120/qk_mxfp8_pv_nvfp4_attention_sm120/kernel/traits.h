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

#include "../common/cute_extension.h"  // Blackwell MMA types
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

namespace qk_mxfp8_pv_nvfp4_attention {

/**
 *
 */
template <int kStages,           // Pipeline stages for K/V
          int EpiStages,         // Epilogue pipeline stages
          typename Element,      // QK element type (float_e4m3_t / FP8)
          typename ElementSF,    // QK scale factor type (float_ue8m0_t)
          typename ElementV,     // V element type (float_e2m1_t / FP4)
          typename ElementSFPV,  // V scale factor type (float_ue4m3_t)
          typename OutputType,   // Output type (bfloat16_t)
          typename ElementDS,    // Delta_s storage type
          typename SmemLayoutQ, typename SmemLayoutK, typename SmemLayoutV,
          typename SmemLayoutDS,  // Delta_s layout
          typename SmemLayoutO, typename SmemLayoutSFQ, typename SmemLayoutSFK,
          typename SmemLayoutSFV>
struct SharedStorageQKVOwithSF : cute::aligned_struct<128, _0> {
  // Q and O share raw smem (Q used during mainloop, O during epilogue)
  static constexpr int kQBytes = sizeof(cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>);
  static constexpr int kOBytes = sizeof(cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>>);
  static constexpr int kQOBytes = kQBytes > kOBytes ? kQBytes : kOBytes;
  alignas(1024) char smem_qo[kQOBytes];

  // Accessors — reinterpret the shared raw storage
  CUTE_DEVICE auto& smem_q_storage() {
    return *reinterpret_cast<cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>>*>(smem_qo);
  }
  CUTE_DEVICE auto& smem_o_storage() {
    return *reinterpret_cast<cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>>*>(smem_qo);
  }

  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> smem_k;

  // Scale factors for Q, K (FP8 path: UE8M0)
  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_SFQ;
  cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_SFK;
  // Scale factors for V (FP4 path: UE4M3, stays FP4 since PV uses NVFP4)
  cute::ArrayEngine<ElementSFPV, cute::cosize_v<SmemLayoutSFV>> smem_SFV;

  alignas(1024) cute::ArrayEngine<ElementV, cute::cosize_v<SmemLayoutV>> smem_v;

  struct {
    alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
    alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
    alignas(16) typename qk_mxfp8_pv_nvfp4_attention::OrderedSequenceBarrierVarGroupSize<
        EpiStages, 2>::SharedStorage barrier_o;
    int tile_count_semaphore;
  };
};

/**
 * Flash Attention Forward Kernel Traits
 *
 * - Pipeline stages
 * - Shared memory layouts
 *
 * @param kStages_: Pipeline stages for K/V
 * @param kClusterM_: Cluster size in M dimension
 * @param ElementPairType_: FP4 pair type
 * @param ElementOut_: Output type (bfloat16/float16)
 */
template <int kHeadDim_, int kBlockM_, int kBlockN_, int kStages_, int kClusterM_, bool BlockMean_,
          typename ElementPairType_ = cutlass::nv_float4_t<cutlass::float_e2m1_t>,
          typename ElementOut_ = cutlass::bfloat16_t, typename ElementDS_ = float>
struct Flash_fwd_kernel_traits {
  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr bool BlockMean = BlockMean_;
  static constexpr bool SmoothQ = true;

  static_assert(kHeadDim % 32 == 0, "Head dim must be multiple of 32");
  static_assert(kBlockM == 64 || kBlockM == 128, "BlockM must be 64 or 128");

  static constexpr int kNWarps = kBlockM == 128 ? 12 : 8;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
  static constexpr int kBlockMPerWG = kBlockM / 2;  // 64
  static constexpr int kNumConsumerWarGroups = 2;

  static constexpr int kClusterM = kClusterM_;
  static constexpr int kStages = kStages_;
  static constexpr int EpiStages = 1;  // Epilogue stages

  static constexpr int NumSFQK = kHeadDim / 32;
  static constexpr auto SFVectorSizeQK = 32;  // QK SF vector size (FP8 mxf8f6f4)
  static constexpr int NumSFPV = kBlockN / 16;
  static constexpr auto SFVectorSizePV = 16;  // PV SF vector size (FP4 mxf4nvf4)
  // Legacy alias (used by PV path code which stays FP4)
  static constexpr auto SFVectorSize = SFVectorSizePV;
  // SM120 FP8 block-scaled MMA stores scale factors in a 128-row/column
  // swizzle atom.  A 64-row compute tile therefore loads one 128-row SFQ
  // tile and selects the appropriate half in the consumer.
  static constexpr int kFP8ScaleBlockMN = 128;
  static constexpr int kSFQScaleBlockM = kBlockM < kFP8ScaleBlockMN ? kFP8ScaleBlockMN : kBlockM;
  static constexpr int kSFKScaleBlockN = kBlockN < kFP8ScaleBlockMN ? kFP8ScaleBlockMN : kBlockN;
  static constexpr int kSFQTilesPerScaleTile = kSFQScaleBlockM / kBlockM;
  static constexpr int kSFKTilesPerScaleTile = kSFKScaleBlockN / kBlockN;

  // QK path: FP8 E4M3 + UE8M0 scale factors
  using Element = cutlass::float_e4m3_t;     // Q/K data: E4M3 (FP8)
  using ElementSF = cutlass::float_ue8m0_t;  // Q/K scale factor: UE8M0
  // PV path: FP4 E2M1 + UE4M3 scale factors (stays FP4 for maximum PV throughput)
  using ElementPV = cutlass::float_e2m1_t;     // P/V data: E2M1 (FP4)
  using ElementSFPV = cutlass::float_ue4m3_t;  // P/V scale factor: UE4M3
                                               // Common
  using ElementAccum = float;                  // Accumulator: FP32
  using ElementOut = ElementOut_;              // Output: BF16/FP16
  using ElementDS = ElementDS_;                // Delta_s storage: FP32 or BF16
  using index_t = int64_t;

  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using SFQTileShape_MNK = Shape<Int<kSFQScaleBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using SFKTileShape_MNK = Shape<Int<kBlockM>, Int<kSFKScaleBlockN>, Int<kHeadDim>>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  using PermTileM = Int<kBlockMPerWG>;  // 64 (WS) or 128 (Non-WS)
  using PermTileN = _32;
  using PermTileK = Int<kHeadDim>;

  using ElementQMma =
      decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<
               Element>());
  using ElementKMma =
      decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<
               Element>());

  // WS or SPLIT_Q: 4 atoms in M per group (4 warps × 16M/atom = 64M)
  using AtomLayoutMNK = Layout<Shape<_4, _1, _1>>;

  // Q@K GEMM: FP8×FP8 (mxf8f6f4, K=32) — composite atom N=32
  using TiledMmaQK =
      decltype(cute::make_tiled_mma(cute::SM120::BLOCKSCALED::SM120_16x32x32_TN_VS_FP8{},
                                    AtomLayoutMNK{}, Tile<PermTileM, PermTileN, PermTileK>{}));

  // P@V GEMM: stays FP4×FP4 (mxf4nvf4, K=64) for maximum throughput.
  using TiledMmaPV =
      decltype(cute::make_tiled_mma(cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
                                    AtomLayoutMNK{}, Tile<PermTileM, _32, PermTileK>{}));

  // Full 128-row MMA (8 atoms, 256 threads).
  // Used for SFQ partitioning and epilogue O store.
  // With consumer_thread_idx 0-255, WG1 maps to atoms 0-3 (rows 0-63),
  // WG2 maps to atoms 4-7 (rows 64-127).
  using AtomLayoutMNK_Full = Layout<Shape<Int<kNumConsumerWarGroups * 4>, _1, _1>>;
  using TiledMmaQK_Full = decltype(cute::make_tiled_mma(
      cute::SM120::BLOCKSCALED::SM120_16x32x32_TN_VS_FP8{}, AtomLayoutMNK_Full{},
      Tile<Int<kBlockM>, PermTileN, PermTileK>{}));
  using TiledMmaPV_Full =
      decltype(cute::make_tiled_mma(cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},
                                    AtomLayoutMNK_Full{}, Tile<Int<kBlockM>, _32, PermTileK>{}));

  static constexpr int MMA_NSF_QK = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVectorSizeQK;
  static constexpr int MMA_NSF_PV = size<2>(typename TiledMmaPV::AtomShape_MNK{}) / SFVectorSizePV;
  // Legacy alias (QK path)
  static constexpr int MMA_NSF = MMA_NSF_QK;

  using GmemTiledCopy = SM90_TMA_LOAD;
  using GmemTiledCopySF = SM90_TMA_LOAD;

  // ============ Shared Memory Layouts ============

  // Q/K smem: use uint8_t for selector (matching CUTLASS blockscaled builder's SmemAllocType)
  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
                                   uint8_t, decltype(size<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
                                   uint8_t, decltype(size<2>(TileShape_MNK{}))>());
  // V smem: FP4 layout (PV GEMM uses NVFP4 atom)
  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
                                   ElementPV, decltype(size<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<
                                    ElementPV, decltype(size<1>(TileShape_MNK{}))>());

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

  // --- Delta_s layout ---
  using SmemLayoutAtomDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<_0, _1>>;
  using SmemLayoutDS = decltype(tile_to_shape(
      SmemLayoutAtomDS{},
      make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

  // ============ Shared Memory Copy Atoms ============
  // Q/K: FP8 data — use ldmatrix (SM75_U32x4_LDSM_N) matching CUTLASS blockscaled builder
  using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomK = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomKV = SmemCopyAtomK;                                   // legacy alias
  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;  // QK SF: UE8M0
  // V: FP4 data, UE4M3 scale factors (PV stays NVFP4)
  using SmemCopyAtomV = Copy_Atom<SM75_U32x4_LDSM_N, ElementPV>;
  using SmemCopyAtomSFPV = Copy_Atom<UniversalCopy<ElementSFPV>, ElementSFPV>;  // PV SF: UE4M3
  using SmemCopyAtomDS = Copy_Atom<UniversalCopy<ElementDS>, ElementDS>;

  // ============ Scale Factor Layouts ============
  using BlkScaledConfigQK =
      qk_mxfp8_pv_nvfp4_attention::BlockScaledConfig<SFVectorSizeQK>;  // FP8 QK, SFVecSize=32
  using BlkScaledConfigPV =
      qk_mxfp8_pv_nvfp4_attention::BlockScaledConfig<SFVectorSizePV>;  // FP4 PV, SFVecSize=16
  // Legacy aliases (QK path is the default)
  using BlkScaledConfig = BlkScaledConfigQK;
  using LayoutSF = typename BlkScaledConfigQK::LayoutSF;    // QK path SF layout (FP8)
  using LayoutSFPV = typename BlkScaledConfigPV::LayoutSF;  // PV path SF layout (FP4)
  using SfAtom = typename BlkScaledConfigQK::SfAtom;

  using SmemLayoutAtomSFQ =
      decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, SFQTileShape_MNK{}));

  using SmemLayoutAtomSFK =
      decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, SFKTileShape_MNK{}));

  // V SF layout: uses PV config (FP4, SFVecSize=16) since PV stays NVFP4
  using SmemLayoutAtomSFV =
      decltype(BlkScaledConfigPV::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
  using SmemLayoutAtomSFVt = decltype(BlkScaledConfigPV::deduce_smem_layoutSFVt(
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

  using SmemLayoutAtomO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementOut, decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{}), Step<_1, _2>{}));
  // Per-WG half of O (64 M-rows × kHeadDim) for ping-pong mma_store
  using SmemLayoutO_Half = decltype(tile_to_shape(
      SmemLayoutAtomO{}, make_shape(Int<kBlockMPerWG>{}, Int<kHeadDim>{}), Step<_1, _2>{}));

  using SharedStorage = SharedStorageQKVOwithSF<kStages, EpiStages, Element, ElementSF, ElementPV,
                                                ElementSFPV, ElementOut, ElementDS, SmemLayoutQ,
                                                SmemLayoutK, SmemLayoutV, SmemLayoutDS, SmemLayoutO,
                                                SmemLayoutSFQ, SmemLayoutSFK, SmemLayoutSFVt>;

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using PipelineState = typename cutlass::PipelineState<kStages>;

  using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
  using PipelineParamsQ = typename MainloopPipelineQ::Params;
  using PipelineStateQ = typename cutlass::PipelineState<1>;

  // Epilogue barrier
  using EpilogueBarrier =
      typename qk_mxfp8_pv_nvfp4_attention::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;

  // Ping-pong math order barrier between Consumer0 and Consumer1
};

}  // namespace qk_mxfp8_pv_nvfp4_attention
