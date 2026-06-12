/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass_extensions/detail/collective/mixed_input_utils.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <int Stages, class ClusterShape, class KernelSchedule_, class TileShape_,
          class ElementAOptionalTuple, class StrideA_, class ElementBOptionalTuple, class StrideB_,
          class TiledMma_, class GmemTiledCopyA_, class SmemLayoutAtomA_, class SmemCopyAtomA_,
          class TransformA_, class GmemTiledCopyB_, class SmemLayoutAtomB_, class SmemCopyAtomB_,
          class TransformB_>
struct CollectiveMmaArrayMixedInput<
    MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>,
    TileShape_, ElementAOptionalTuple, StrideA_, ElementBOptionalTuple, StrideB_, TiledMma_,
    GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_,
    SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_> {
 public:
  enum class ConversionMode { DirectConvert, ConvertAndScale };

  //
  // Type Aliases
  //
  using DispatchPolicy =
      MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape, KernelSchedule_>;
  using TileShape = TileShape_;
  using KernelSchedule = KernelSchedule_;

 private:
  template <class T>
  friend struct detail::MixedGroupedGemmInputUtils;
  using CollectiveType =
      CollectiveMmaArrayMixedInput<DispatchPolicy, TileShape_, ElementAOptionalTuple, StrideA_,
                                   ElementBOptionalTuple, StrideB_, TiledMma_, GmemTiledCopyA_,
                                   SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_,
                                   SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_>;
  using Utils = detail::MixedGroupedGemmInputUtils<CollectiveType>;

  //
  // Type Aliases
  //
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;

 public:
  static_assert(cute::is_tuple<ElementAOptionalTuple>::value ^
                    cute::is_tuple<ElementBOptionalTuple>::value,
                "Either A OR B must be a tuple. It must take the from {ElementOperand, "
                "[ElementScale], [ElementZero]}. Inputs "
                "in [] are optional.");

  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed = cute::is_tuple<ElementAOptionalTuple>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;
  // For cases where we can't have a void type, we can use this to allow the code to compile when
  // the scale / zero is void.
  using NonVoidElementScale =
      cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;

  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using NonVoidStrideScale = cute::conditional_t<cute::is_void_v<StrideScale>,
                                                 cute::Stride<_1, int64_t, int64_t>, StrideScale>;

  static_assert(
      (IsATransformed && (cutlass::gemm::detail::is_k_major<StrideA>() ||
                          is_layout<StrideA>::value || is_layout<InternalStrideA>::value)) ||
          (!IsATransformed && (cutlass::gemm::detail::is_k_major<StrideB>() ||
                               is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),
      "The transformed type must be K-major.");

  static_assert((IsATransformed && (sizeof(ElementB) == 2)) ||
                    (!IsATransformed && (sizeof(ElementA) == 2)) ||
                    ((cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value ||
                      is_layout<InternalStrideA>::value) &&
                     (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value ||
                      is_layout<InternalStrideB>::value)),
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(cutlass::gemm::detail::is_mn_major<NonVoidStrideScale>(),
                "Scale must be MN major [Col Major if A is scaled, Row Major if B is scaled].");

  static constexpr bool IsMXFP4 = cute::is_same_v<ElementA, cutlass::float_e2m1_t>;
  static constexpr bool IsInt4Weight = cute::is_same_v<ElementA, cutlass::int4b_t>;
  static_assert(IsMXFP4 || IsInt4Weight,
                "Folded weight scale is only defined for MXFP4 and INT4 weights.");
  static constexpr int ScalingGroupSize = detail::DefaultWeightScaleGroupSize<ElementA>::value;

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = !IsATransformed;
  using SwappedStrideA = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using SwappedStrideB = cute::conditional_t<!SwapAB, StrideB, StrideA>;
  using InternalSwappedStrideA = cute::conditional_t<!SwapAB, InternalStrideA, InternalStrideB>;
  using InternalSwappedStrideB = cute::conditional_t<!SwapAB, InternalStrideB, InternalStrideA>;
  using SwappedSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using SwappedSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using SwappedSmemCopyAtomA = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using SwappedSmemCopyAtomB = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using ConvertedElementA =
      cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB =
      cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  using RealSwappedElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealSwappedElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using SwappedElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using SwappedElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using SwappedTransformA = cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using SwappedTransformB = cute::conditional_t<!SwapAB, TransformB, TransformA>;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int IsSubbyteA = cute::sizeof_bits_v<SwappedElementA> < 8;
  using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  static constexpr int NumProducerThreadEvents = 1;

  static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2,
                "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SwappedSmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2,
                "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SwappedSmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  /// Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(
      SwappedSmemLayoutAtomA{}, select<0, 2>(TileShape{}), InternalSwappedStrideA{}));
  using SmemLayoutB = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(
      SwappedSmemLayoutAtomB{}, select<1, 2>(TileShape{}), InternalSwappedStrideB{}));

  using WeightScaleRawElement = NonVoidElementScale;
  static_assert(!cutlass::detail::is_Array_v<NonVoidElementScale>,
                "Folded weight scale uses scalar scale storage; packed scale arrays are not supported.");
  static_assert(cute::is_void_v<ElementZero>,
                "Folded weight scale storage does not support zero-point.");
  static constexpr int WeightScaleLogicalMPerFoldBlock = 64;
  static constexpr int WeightScaleLogicalKPerFoldBlock = 128;
  static constexpr int WeightScalePhysicalColsPerFoldBlock =
      128 / cutlass::sizeof_bits<WeightScaleRawElement>::value;
  static constexpr int WeightScaleScaleGroupsPerFoldBlock =
      WeightScaleLogicalKPerFoldBlock / ScalingGroupSize;
  static constexpr int WeightScaleMSlicesPerFoldBlock =
      WeightScalePhysicalColsPerFoldBlock / WeightScaleScaleGroupsPerFoldBlock;
  static constexpr int WeightScaleFoldedMPerFoldBlock =
      WeightScaleLogicalMPerFoldBlock / WeightScaleMSlicesPerFoldBlock;
  static constexpr int WeightScaleMBlocksPerTile =
      size<0>(TileShape{}) / WeightScaleLogicalMPerFoldBlock;
  static constexpr int WeightScaleKBlocksPerTile =
      size<2>(TileShape{}) / WeightScaleLogicalKPerFoldBlock;
  static_assert(size<0>(TileShape{}) % WeightScaleLogicalMPerFoldBlock == 0,
                "Folded weight scale requires TileShapeM to be a multiple of 64.");
  static_assert(size<2>(TileShape{}) % WeightScaleLogicalKPerFoldBlock == 0,
                "Folded weight scale requires TileShapeK to be a multiple of 128.");
  static_assert(WeightScalePhysicalColsPerFoldBlock % WeightScaleScaleGroupsPerFoldBlock == 0,
                "Folded weight scale requires each M slice to contain an integer number of scale "
                "groups.");
  static_assert(WeightScaleLogicalMPerFoldBlock % WeightScaleMSlicesPerFoldBlock == 0,
                "Folded weight scale M slices must evenly divide the logical M block.");
  static_assert(WeightScalePhysicalColsPerFoldBlock *
                        cutlass::sizeof_bits<WeightScaleRawElement>::value ==
                    128,
                "Folded weight scale must expose 16B per folded-M coordinate.");
  static constexpr int WeightScaleRawElementsPerFoldBlock =
      WeightScaleLogicalMPerFoldBlock * WeightScaleLogicalKPerFoldBlock / ScalingGroupSize;
  static constexpr int WeightScaleRawElementsPerStage =
      WeightScaleRawElementsPerFoldBlock * WeightScaleMBlocksPerTile * WeightScaleKBlocksPerTile;
  static constexpr uint32_t WeightScaleFoldBlockBytes =
      cutlass::bits_to_bytes(WeightScaleRawElementsPerFoldBlock *
                             cutlass::sizeof_bits<WeightScaleRawElement>::value);
  static constexpr uint32_t WeightScaleBulkCopyBytes =
      WeightScaleFoldBlockBytes * WeightScaleKBlocksPerTile;
  static constexpr uint32_t WeightScaleTransactionBytes =
      cutlass::bits_to_bytes(WeightScaleRawElementsPerStage *
                             cutlass::sizeof_bits<WeightScaleRawElement>::value);
  static_assert(WeightScaleBulkCopyBytes % 16 == 0,
                "Folded weight-scale bulk copy size must be 16B aligned.");

  static constexpr bool IsInt4Fp8Path =
      cute::is_same_v<ElementA, cutlass::int4b_t> &&
      cute::is_same_v<ElementB, cutlass::float_e4m3_t>;
  static constexpr bool IsMxfp4Bf16Path =
      IsMXFP4 && cute::is_same_v<ElementB, cutlass::bfloat16_t>;
  static constexpr bool IsMxfp4Fp8Path =
      IsMXFP4 && cute::is_same_v<ElementB, cutlass::float_e4m3_t>;

  static constexpr bool UseDirectSmemWeightScale =
      (
        (IsMxfp4Fp8Path &&
         ((size<0>(TileShape{}) == 64 && size<1>(TileShape{}) == 64 &&
           size<0>(ClusterShape{}) == 1 && size<1>(ClusterShape{}) == 1) ||
          (size<0>(TileShape{}) == 64 && size<1>(TileShape{}) == 128 &&
           size<2>(TileShape{}) == 256 && size<0>(ClusterShape{}) == 1 &&
           size<1>(ClusterShape{}) == 1) ||
          (size<0>(TileShape{}) == 128 && size<1>(TileShape{}) == 256 &&
           size<2>(TileShape{}) == 256) ||
          (size<0>(TileShape{}) == 256 && size<1>(TileShape{}) == 128 &&
           size<2>(TileShape{}) == 256) ||
          (size<0>(TileShape{}) == 128 && size<1>(TileShape{}) == 64 &&
           size<2>(TileShape{}) == 512 &&
           (size<0>(ClusterShape{}) != 1 || size<1>(ClusterShape{}) != 1)))) ||
        (IsMxfp4Bf16Path &&
         ((size<0>(TileShape{}) == 64 && size<1>(TileShape{}) == 64 &&
           size<0>(ClusterShape{}) == 1 && size<1>(ClusterShape{}) == 1) ||
          (size<0>(TileShape{}) == 64 && size<1>(TileShape{}) == 128 &&
           size<2>(TileShape{}) == 256 && size<0>(ClusterShape{}) == 1 &&
           size<1>(ClusterShape{}) == 1))) ||
        (IsInt4Fp8Path &&
         ((size<0>(TileShape{}) == 64 && size<1>(TileShape{}) == 64 &&
           size<2>(TileShape{}) == 256 && size<0>(ClusterShape{}) == 1 &&
           size<1>(ClusterShape{}) == 1) ||
          (size<0>(TileShape{}) == 128 && size<1>(TileShape{}) == 64 &&
           size<2>(TileShape{}) == 512 && size<0>(ClusterShape{}) == 1 &&
           size<1>(ClusterShape{}) == 2)))
      );
  using SmemLayoutWeightScaleRaw = Layout<
      Shape<Int<WeightScalePhysicalColsPerFoldBlock>, Int<WeightScaleFoldedMPerFoldBlock>,
            Int<WeightScaleMBlocksPerTile>, Int<WeightScaleKBlocksPerTile>, Int<Stages>>,
      Stride<_1, Int<WeightScalePhysicalColsPerFoldBlock>,
             Int<WeightScaleFoldedMPerFoldBlock * WeightScalePhysicalColsPerFoldBlock *
                 WeightScaleKBlocksPerTile>,
             Int<WeightScaleFoldedMPerFoldBlock * WeightScalePhysicalColsPerFoldBlock>,
             Int<WeightScaleRawElementsPerStage>>>;
  using SmemLayoutWeightScaleExpanded = Layout<
      Shape<Shape<Int<WeightScaleFoldedMPerFoldBlock>, Int<WeightScaleMSlicesPerFoldBlock>,
                  Int<WeightScaleMBlocksPerTile>>,
            Shape<Int<ScalingGroupSize>,
                  Shape<Int<WeightScaleScaleGroupsPerFoldBlock>,
                        Int<WeightScaleKBlocksPerTile>>>,
            Int<Stages>>,
      Stride<Stride<Int<WeightScalePhysicalColsPerFoldBlock>,
                    Int<WeightScaleScaleGroupsPerFoldBlock>,
                    Int<WeightScaleFoldedMPerFoldBlock *
                        WeightScalePhysicalColsPerFoldBlock * WeightScaleKBlocksPerTile>>,
             Stride<_0, Stride<_1,
                                Int<WeightScaleFoldedMPerFoldBlock *
                                    WeightScalePhysicalColsPerFoldBlock>>>,
             Int<WeightScaleRawElementsPerStage>>>;

  static_assert(DispatchPolicy::Stages >= 2,
                "Specialization requires Stages set to value 2 or more.");
  static_assert(
      not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
          cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
      "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> ||
                    cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
                "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> ||
                    cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
                "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

 private:
  static constexpr ConversionMode get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    } else {
      return ConversionMode::ConvertAndScale;
    }
  }

  bool TensormapUpdateShapesStridesForAandScale = true;

 public:
  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales =
      KernelConversionMode == ConversionMode::ConvertAndScale;
  static constexpr bool UseScaleLookupTable =
      KernelConversionMode == ConversionMode::ConvertAndScale &&
      cutlass::detail::is_Array_v<ElementScale>;
  static constexpr bool UseFP4ToBF16LookupTable =
      KernelConversionMode == ConversionMode::ConvertAndScale &&
      cute::is_same_v<ElementA, cutlass::float_e2m1_t> &&
      cute::is_same_v<ElementB, cutlass::bfloat16_t>;
  static constexpr bool UseInt4ToFP8LookupTable =
      KernelConversionMode == ConversionMode::ConvertAndScale &&
      cute::is_same_v<ElementA, cutlass::int4_t> &&
      cute::is_same_v<ElementB, cutlass::float_e4m3_t>;
  static constexpr size_t SmemAlignmentA = cutlass::detail::alignment_for_swizzle(SmemLayoutA{});
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});
  static constexpr size_t SmemAlignmentScale = cute::max(SmemAlignmentA, SmemAlignmentB);

  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");

  struct SharedStorage {
    static constexpr int scale_elements = cute::cosize_v<SmemLayoutWeightScaleRaw>;

    struct TensorStorage {
      CUTE_ALIGNAS(SmemAlignmentA)
      cute::ArrayEngine<RealSwappedElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      CUTE_ALIGNAS(SmemAlignmentB)
      cute::ArrayEngine<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
      cute::ArrayEngine<WeightScaleRawElement, scale_elements> smem_scale;
    } tensors;

    struct TensorMapStorage {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
    };

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };

  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // kernel Arguments
  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    ElementScale const** ptr_S = nullptr;
    NonVoidStrideScale const* dS{};
    int chunk_size = 0;
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using LayoutA = decltype(detail::get_gmem_layout(
        repeat_like(InternalSwappedStrideA{}, int32_t(0)), InternalSwappedStrideA{}));
    using LayoutB = decltype(detail::get_gmem_layout(
        repeat_like(InternalSwappedStrideB{}, int32_t(0)), InternalSwappedStrideB{}));

    using TMA_A = decltype(make_tma_copy<TmaElementA>(
        GmemTiledCopyA{},
        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementA const*>(nullptr)),
                    LayoutA{}),
        SmemLayoutA{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{})));  // mcast along N mode for this M load, if any
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(detail::get_logical_ptr(static_cast<SwappedElementB const*>(nullptr)),
                    LayoutB{}),
        SmemLayoutB{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})));  // mcast along M mode for this N load, if any

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    void* tensormaps;
    SwappedElementA const** ptr_A;
    SwappedStrideA ptr_dA;
    SwappedElementB const** ptr_B;
    SwappedStrideB ptr_dB;
    NonVoidElementScale const** ptr_S;
    NonVoidStrideScale const* dS;
    int chunk_size;
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape problem_shapes,
                                                  Arguments const& args, void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create
    // tensormap/tma desc. These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);

    if constexpr (SwapAB) {
      init_M = get<1>(init_shape);
      init_N = get<0>(init_shape);
    }
    // Batches/Groups are managed by using appropriate pointers to input matrices
    const uint32_t mock_L = 1;
    SwappedElementA const* ptr_A_first_batch;
    SwappedElementB const* ptr_B_first_batch;
    SwappedStrideA ptr_dA;
    SwappedStrideB ptr_dB;
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;

    if constexpr (not SwapAB) {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_A);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_B);
    } else {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_B);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_A);
    }

    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      if constexpr (not SwapAB) {
        ptr_dA = args.dA;
        ptr_dB = args.dB;
      } else {
        ptr_dA = args.dB;
        ptr_dB = args.dA;
      }
      dA = InternalSwappedStrideA{};
      if constexpr (is_layout<InternalSwappedStrideA>::value) {
        dA = make_layout(transform_leaf(dA.shape(),
                                        [](auto x) {
                                          if constexpr (not is_static_v<decltype(x)>) {
                                            return static_cast<decltype(x)>(1);
                                          } else {
                                            return x;
                                          }
                                        }),
                         dA.stride());
      }
      dB = InternalSwappedStrideB{};
    } else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      if constexpr (not SwapAB) {
        dA = args.dA;
        dB = args.dB;
      } else {
        dA = args.dB;
        dB = args.dA;
      }
      ptr_dA = SwappedStrideA{};
      ptr_dB = SwappedStrideB{};
    }
    Tensor tensor_a = make_tensor(ptr_A_first_batch,
                                  detail::get_gmem_layout(make_shape(init_M, init_K, mock_L), dA));
    Tensor tensor_b = make_tensor(ptr_B_first_batch,
                                  detail::get_gmem_layout(make_shape(init_N, init_K, mock_L), dB));

    typename Params::TMA_A tma_load_a = make_tma_copy<TmaElementA>(
        GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{}));  // mcast along N mode for this M load, if any
    typename Params::TMA_B tma_load_b =
        make_tma_copy(GmemTiledCopyB{}, tensor_b, SmemLayoutB{}(_, _, cute::Int<0>{}),
                      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
                      size<0>(ClusterShape{}));  // mcast along M mode for this N load, if any
    void* tensormaps = workspace;
    auto args_setup = [&](auto ptr_A, auto ptr_B, int chunk_size = 0) -> Params {
      return {tma_load_a,
              tma_load_b,
              TmaTransactionBytes,
              tensormaps,
              reinterpret_cast<SwappedElementA const**>(ptr_A),
              ptr_dA,
              reinterpret_cast<SwappedElementB const**>(ptr_B),
              ptr_dB,
              reinterpret_cast<NonVoidElementScale const**>(args.ptr_S),
              args.dS,
              chunk_size,
              dA,
              dB};
    };

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return SwapAB ? args_setup(args.ptr_B, args.ptr_A) : args_setup(args.ptr_A, args.ptr_B);
    } else if constexpr (ModeHasScales) {
      return SwapAB ? args_setup(args.ptr_B, args.ptr_A, args.chunk_size)
                    : args_setup(args.ptr_A, args.ptr_B, args.chunk_size);
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in to_underlying_arguments.");
    }
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args,
                                   int sm_count) {
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);

    // Calculating workspace size
    auto calculate_workspace_size = [SizeOfCuTensorMap, sm_count](uint32_t num_input_tensors) {
      return num_input_tensors * SizeOfCuTensorMap * sm_count;
    };

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // Allocate gmem space for input tensormaps per each SM, A tensormap copies followed by B
      // tensormap copies
      return calculate_workspace_size(2);
    } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      return calculate_workspace_size(2);
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in get_workspace_size.");
    }
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape,
                                              Arguments const& args, void* workspace,
                                              cudaStream_t stream,
                                              CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(ProblemShape problem_shapes,
                                                Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A =
        tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B =
        tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M, N, K, L] = problem_shape_MNKL;
        auto get_stride = [](auto stride) {
          if constexpr (cute::is_pointer_v<cute::decay_t<decltype(stride)>>) {
            return *stride;
          } else {
            return stride;
          }
        };
        auto dA = get_stride(args.dA);
        auto dB = get_stride(args.dB);
        implementable =
            implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(
                                 detail::get_gmem_layout(cute::make_shape(M, K, L), dA));
        implementable =
            implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(
                                 detail::get_gmem_layout(cute::make_shape(N, K, L), dB));
        if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
          implementable = implementable && (args.ptr_S == nullptr);
        } else if constexpr (ModeHasScales) {
          int const scale_mn = SwapAB ? N : M;
          implementable = implementable && args.chunk_size != 0;
          implementable = implementable && (args.ptr_S != nullptr);
          implementable = implementable && (args.chunk_size == ScalingGroupSize);
          implementable = implementable && ((scale_mn % size<0>(TileShape{})) == 0);
          implementable = implementable && ((K % size<2>(TileShape{})) == 0);
          implementable = implementable && ((K % WeightScaleLogicalKPerFoldBlock) == 0);
        } else {
          static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                        "Conversion mode not handled in can_implement.");
        }
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for "
          "TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK = Utils::compute_tma_transaction_bytes_mk();
  static constexpr uint32_t TmaTransactionBytesNK = Utils::compute_tma_transaction_bytes_nk();
  static constexpr uint32_t TmaTransactionBytesExtra = Utils::compute_tma_transaction_bytes_extra();
  static constexpr uint32_t TmaTransactionBytes =
      TmaTransactionBytesMK + TmaTransactionBytesNK + TmaTransactionBytesExtra;

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  // The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL,
                                Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;
    const int32_t mock_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(
        shape(detail::get_gmem_layout(make_shape(M, K, mock_L), mainloop_params.dA)));  // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(
        shape(detail::get_gmem_layout(make_shape(N, K, mock_L), mainloop_params.dB)));  // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_, _, _),
                               Step<_1, X, _1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_, _, _),
                               Step<X, _1, _1>{});  // (BLK_N,BLK_K,n,k,l)
    int const scale_total_k128_blocks = int(K) / WeightScaleLogicalKPerFoldBlock;

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(gA_mkl, gB_nkl);
    } else if constexpr (ModeHasScales) {
      return cute::make_tuple(gA_mkl, gB_nkl, static_cast<NonVoidElementScale const*>(nullptr),
                              int64_t(0), scale_total_k128_blocks);
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in load_init.");
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <class... Ts, class... TMs, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline,
                           PipelineState smem_pipe_write, cute::tuple<Ts...> const& load_inputs,
                           cute::tuple<TMs...> const& input_tensormaps, BlockCoord const& blk_coord,
                           KTileIterator k_tile_iter, int k_tile_count, int thread_idx,
                           uint32_t block_rank_in_cluster, TensorStorage& shared_tensors) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      static_assert(sizeof...(Ts) == 2, "Direct convert needs two inputs");
      static_assert(sizeof...(TMs) == 2, "Direct convert needs two tensormaps");
    } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      static_assert(sizeof...(Ts) == 5, "Folded scaled convert needs five inputs");
      static_assert(sizeof...(TMs) == 2, "Folded scaled convert needs two tensormaps");
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in TMA load.");
    }

    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                             SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                             SmemLayoutB{});                  // (BLK_N,BLK_K,PIPE)
    Tensor sA = as_position_independent_swizzle_tensor(sA_);  // (BLK_M,BLK_K,PIPE)
    Tensor sB = as_position_independent_swizzle_tensor(sB_);  // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for A and B
    //

    constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                    block_rank_in_cluster / cluster_shape_x};

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);  // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);  // (BLK_N,BLK_K,k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);  // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);  // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);  // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);  // (TMA,TMA_N,TMA_K,PIPE)

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;

    // Issue TmaLoads
    // Maps the tile -> block, value
    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};  // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, Int<0>{}));
      }
    }

    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};  // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_b |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, Int<0>{}));
      }
    }

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (cute::elect_one_sync()) {
        copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a),
             tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b),
             tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
      }
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // Nothing extra to do.
      } else if constexpr (ModeHasScales) {
        if (cute::elect_one_sync()) {
          auto scale_ptr = get<2>(load_inputs);
          int const scale_k_tile = *k_tile_iter;
          int const scale_total_k128_blocks = get<4>(load_inputs);
          int const scale_k128_offset = scale_k_tile * WeightScaleKBlocksPerTile;
          int const scale_m64_offset =
              int(m_coord) * int(size<0>(TileShape{})) / WeightScaleLogicalMPerFoldBlock;
          auto* scale_base = reinterpret_cast<WeightScaleRawElement const*>(scale_ptr);
          Tensor sSRaw = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()),
                                     SmemLayoutWeightScaleRaw{});

          CUTLASS_PRAGMA_UNROLL
          for (int local_m64_block = 0; local_m64_block < WeightScaleMBlocksPerTile;
               ++local_m64_block) {
            int const m64_block = scale_m64_offset + local_m64_block;
            int64_t const scale_gmem_fold_block =
                int64_t(m64_block) * int64_t(scale_total_k128_blocks) +
                int64_t(scale_k128_offset);
            int64_t const scale_gmem_offset =
                scale_gmem_fold_block * int64_t(WeightScaleRawElementsPerFoldBlock);
            auto* scale_gmem_addr = reinterpret_cast<void const*>(scale_base + scale_gmem_offset);
            auto* scale_smem_addr =
                static_cast<void*>(&sSRaw(0, 0, local_m64_block, 0, write_stage));
            cute::SM90_BULK_COPY_G2S::copy(
                scale_gmem_addr, reinterpret_cast<uint64_t*>(tma_barrier), scale_smem_addr,
                WeightScaleBulkCopyBytes);
          }
        }
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled for TMA copy op.");
      }
      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <class T>
  CUTLASS_DEVICE float scale_convertor(T scale) {
    if constexpr (cute::is_same_v<ElementA, cutlass::float_e2m1_t>) {
      cutlass::float_ue8m0_t scale_ue8m0 = scale;

      uint32_t temp = static_cast<uint32_t>(scale_ue8m0.storage) << 23;
      return cutlass::detail::copy_bits<uint32_t, float>(temp);
    } else {
      return static_cast<float>(scale);
    }
  }

  template <class ScaleTensor>
  CUTLASS_DEVICE float load_weight_scale(ScaleTensor const& tCrS, int scale_idx, int mma_m,
                                         int m) {
    return scale_convertor(tCrS(make_coord(make_tuple(0, m, 0), mma_m, scale_idx)));
  }

  template <class ScaleTensor>
  CUTLASS_DEVICE float load_weight_scale_smem(ScaleTensor const& tCsS, int scale_idx, int mma_m,
                                              int m, int read_stage) {
    return scale_convertor(tCsS(make_coord(make_tuple(0, m, 0), mma_m, scale_idx, read_stage)));
  }

  template <class AccumTensor, class IntermTensor, class ScaleTensor>
  CUTLASS_DEVICE void apply_groupwise_scale(AccumTensor& accum, IntermTensor const& intermediate,
                                            ScaleTensor const& tCrS, int scale_idx,
                                            bool is_first_accum) {
    multiply_add<ElementAccumulator> fma_op;

    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < size<1>(accum); mma_m++) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<0, 1>(accum); m++) {
        float scale_val = load_weight_scale(tCrS, scale_idx, mma_m, m);
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<0, 2>(accum); n++) {
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < size<0, 0>(accum); e++) {
            auto coord = make_coord(make_tuple(e, m, n), mma_m, 0);
            if (is_first_accum) {
              accum(coord) = intermediate(coord) * scale_val;
            } else {
              accum(coord) = fma_op(intermediate(coord), scale_val, accum(coord));
            }
          }
        }
      }
    }
  }

  template <class AccumTensor, class IntermTensor, class ScaleTensor>
  CUTLASS_DEVICE void apply_groupwise_scale_smem(AccumTensor& accum,
                                                 IntermTensor const& intermediate,
                                                 ScaleTensor const& tCsS, int scale_idx,
                                                 int read_stage, bool is_first_accum) {
    multiply_add<ElementAccumulator> fma_op;

    CUTLASS_PRAGMA_UNROLL
    for (int mma_m = 0; mma_m < size<1>(accum); mma_m++) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<0, 1>(accum); m++) {
        float scale_val = load_weight_scale_smem(tCsS, scale_idx, mma_m, m, read_stage);
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < size<0, 2>(accum); n++) {
          CUTLASS_PRAGMA_UNROLL
          for (int e = 0; e < size<0, 0>(accum); e++) {
            auto coord = make_coord(make_tuple(e, m, n), mma_m, 0);
            if (is_first_accum) {
              accum(coord) = intermediate(coord) * scale_val;
            } else {
              accum(coord) = fma_op(intermediate(coord), scale_val, accum(coord));
            }
          }
        }
      }
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <class FrgTensorC>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                          FrgTensorC& accum, int k_tile_count, int thread_idx,
                          TensorStorage& shared_tensors, Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2,
                  "SwappedSmemLayoutAtomA must be rank 2.");
    static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2,
                  "SwappedSmemLayoutAtomB must be rank 2.");
    static_assert(
        !cute::is_void_v<SwappedSmemCopyAtomA>,
        "SM90 GMMA mainloops must specify a non-void copy atom for smem sourced instructions.");
    static_assert(
        cute::is_void_v<SwappedSmemCopyAtomB>,
        "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    // Obtain warp index
    int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                             SmemLayoutA{});                  // (BLK_M,BLK_K,PIPE)
    Tensor sA = as_position_independent_swizzle_tensor(sA_);  // (BLK_M,BLK_K,PIPE)

    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                            SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and
                      size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be "
                  "NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout =
        make_layout(Int<MmaWarpGroups>{}, Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCsA = mma_thread_slice.partition_A(sA);
    auto mma_warpgroup_slice = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    // Allocate fragments and descriptors
    Tensor tCrA_mma =
        mma_thread_slice.partition_fragment_A(sA(_, _, Int<0>{}));  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrA_load = [&] {
      if constexpr (not is_layout<InternalSwappedStrideA>::value) {
        // Make register tensor with MMA layout
        return make_fragment_like<RealSwappedElementA>(tCrA_mma);
      } else {
        // Make register tensor matching smem layout, converter will take care of de-swizzling
        return make_tensor_like<RealSwappedElementA>(tCsA(_, _, _, Int<0>{}));
      }
    }();
    Tensor tCsB = mma_warpgroup_slice.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)
    // tCrB is just a view of the tensor tCsB
    Tensor tCrB = mma_warpgroup_slice.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)

    //
    // Copy Atom A retiling
    //
    auto smem_tiled_copy_A = make_tiled_copy_A(SwappedSmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);

    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA_load);  // (CPY,CPY_M,CPY_K)

    constexpr int MmaKPerKBlock = cute::get<0, 1>(tCsB.shape())();
    constexpr int NumMMAsPerChunk = ScalingGroupSize / MmaKPerKBlock;
    constexpr int KBlockMaxForScale = size<2>(TileShape{}) / MmaKPerKBlock;
    constexpr int NumChunksPerTileK = cute::size<1>(sA.shape())() / ScalingGroupSize;

    Tensor sS =
        make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()), SmemLayoutWeightScaleExpanded{});
    Tensor tCsS = mma_thread_slice.partition_A(sS);
    Tensor tCrS =
        make_tensor<WeightScaleRawElement>(mma_thread_slice.partition_fragment_A(sS(_, _, Int<0>{})).layout());

    using SmemCopyAtomScaleRaw = Copy_Atom<cute::AutoVectorizingCopy, WeightScaleRawElement>;
    auto smem_tiled_copy_S = make_tiled_copy_A(SmemCopyAtomScaleRaw{}, tiled_mma);
    auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(warp_group_thread_idx);
    Tensor tCrS_copy_view = smem_thr_copy_S.retile_D(tCrS);

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));      // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));      // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));           // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));               // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));  // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));  // PIPE

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    using SmemCopyAtomA_LDSM = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;

    auto smem_tiled_copy_A_LDSM = make_tiled_copy_A(SmemCopyAtomA_LDSM{}, tiled_mma);
    auto smem_thr_copy_A_LDSM = smem_tiled_copy_A_LDSM.get_thread_slice(thread_idx);

    Tensor sA_LDSM = recast<ElementB>(sA);
    auto tCsA_LDSM = smem_thr_copy_A_LDSM.partition_S(sA_LDSM);

    using ABBitWidthRatio = Int<sizeof_bits_v<ElementB> / sizeof_bits_v<ElementA>>;
    auto tCrA_load_LDSM_shape =
        replace<2>(tCrA_mma.shape(), size(get<2>(tCrA_mma.shape())) / ABBitWidthRatio{});
    Tensor tCrA_load_LDSM = make_fragment_like<ElementB>(tCrA_load_LDSM_shape);
    Tensor tCrA_copy_view_LDSM =
        smem_thr_copy_A_LDSM.retile_D(tCrA_load_LDSM);  // (CPY,CPY_M,CPY_K)

    auto ptr = recast_ptr<RealSwappedElementA>(tCrA_load_LDSM.data());
    auto old_shape = tCrA_load_LDSM.shape();
    auto tCrA_load_4b_layout = make_layout(
        make_shape(size<0>(old_shape), get<1>(old_shape),
                   make_shape(ABBitWidthRatio{}, size<2>(old_shape))),
        make_stride(Int<1>{}, size<0>(old_shape) * ABBitWidthRatio{},
                    make_stride(size<0>(old_shape),
                                size<0>(old_shape) * ABBitWidthRatio{} *
                                    size<1>(old_shape))));
    Tensor tCrA_load_4b_packed = make_tensor(ptr, tCrA_load_4b_layout);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    //
    // PIPELINED MAIN LOOP
    //

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    cute::array<decltype(make_fragment_like(accum)), NumChunksPerTileK> intermediate_array;

    constexpr int K_BLOCK_MAX = size<2>(tCrA_load);
    static_assert(K_BLOCK_MAX == KBlockMaxForScale,
                  "Scale and A operand K-block counts must match.");
    constexpr int K_WAIT_MAX = cute::min(K_BLOCK_MAX - 1, 7);
    static_assert(K_BLOCK_MAX >= 4, "Consider increasing TileShapeK");

    auto copy_scale_for_tile = [&](int read_stage) {
      if constexpr (UseDirectSmemWeightScale) {
        (void)read_stage;
      } else {
        if constexpr (NumMMAsPerChunk == 1) {
          cute::for_each(cute::make_seq<KBlockMaxForScale>{}, [&](auto k_block_c) {
            constexpr int k_block = decltype(k_block_c)::value;
            cute::copy(smem_tiled_copy_S, tCsS(_, _, Int<k_block>{}, read_stage),
                       tCrS_copy_view(_, _, Int<k_block>{}));
          });
        } else {
          cute::for_each(cute::make_seq<NumChunksPerTileK>{}, [&](auto scale_idx_c) {
            constexpr int scale_idx = decltype(scale_idx_c)::value;
            constexpr int k_block = scale_idx * NumMMAsPerChunk;
            cute::copy(smem_tiled_copy_S, tCsS(_, _, Int<k_block>{}, read_stage),
                       tCrS_copy_view(_, _, Int<k_block>{}));
          });
        }
      }
    };

    auto scale_intermediate = [&](auto const& intermediate, int scale_idx, int read_stage,
                                  bool is_first_accum) {
      int const weight_scale_idx = scale_idx * NumMMAsPerChunk;
      if constexpr (UseDirectSmemWeightScale) {
        apply_groupwise_scale_smem(accum, intermediate, tCsS, weight_scale_idx, read_stage,
                                   is_first_accum);
      } else {
        apply_groupwise_scale(accum, intermediate, tCrS, weight_scale_idx, is_first_accum);
      }
    };

    ConsumerToken barrier_token = {BarrierStatus::WaitAgain};
    // First k tile
    {
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();

      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

      Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0, read_stage);
      if (K_BLOCK_MAX > 1) {
        Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1,
                              read_stage);
      }

      // src: tCrA_load, dst: tCrA_mma
      Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, cute::Int<0>{});

      // Unroll the K mode manually to set scale D to 1
      cute::for_each(cute::make_seq<NumChunksPerTileK>{}, [&](auto chunk_id_c) {
        constexpr int chunk_id = decltype(chunk_id_c)::value;
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        cute::for_each(cute::make_seq<NumMMAsPerChunk>{}, [&](auto mma_id_c) {
          constexpr int mma_id = decltype(mma_id_c)::value;
          constexpr int k_block = chunk_id * NumMMAsPerChunk + mma_id;

          warpgroup_arrive();

          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma, tCrA_mma(_, _, cute::Int<k_block>{}),
                     tCrB(_, _, cute::Int<k_block>{}, read_stage),
                     intermediate_array[chunk_id]);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;

          if constexpr (k_block == 0) {
            copy_scale_for_tile(read_stage);
          }

          if constexpr (k_block < K_BLOCK_MAX - 2) {
            Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM,
                                  k_block + 2, read_stage);
          }
          if constexpr (k_block < K_BLOCK_MAX - 1) {
            Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma,
                                    cute::Int<k_block + 1>{});
          }
        });

        warpgroup_commit_batch();

        if constexpr (chunk_id > 0) {
          warpgroup_wait<1>();

          constexpr int chunk_id_ = chunk_id - 1;
          warpgroup_fence_operand(intermediate_array[chunk_id_]);

          scale_intermediate(intermediate_array[chunk_id_], chunk_id_, read_stage,
                             chunk_id_ == 0);
        }
      });

      warpgroup_wait<0>();

      constexpr int chunk_id_ = NumChunksPerTileK - 1;
      warpgroup_fence_operand(intermediate_array[chunk_id_]);

      scale_intermediate(intermediate_array[chunk_id_], chunk_id_, read_stage,
                         NumChunksPerTileK == 1);

      --k_tile_count;
      if (k_tile_count > 0) {
        // Wait for K_BLOCK_MAX - 1 to be in flight to ensure that it is safe to overwrite the A
        // registers for the first mma.
        pipeline.consumer_wait(smem_pipe_read, barrier_token);

        Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0,
                              smem_pipe_read.index());
        Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1,
                              smem_pipe_read.index());

        Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, cute::Int<0>{});
      }
    }

    if (k_tile_count == 0) {
      return;
    }

    // Mainloop GMMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 1; --k_tile_count) {
      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;

      // Unroll the K mode manually to set scale D to 1
      cute::for_each(cute::make_seq<NumChunksPerTileK>{}, [&](auto chunk_id_c) {
        constexpr int chunk_id = decltype(chunk_id_c)::value;
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        cute::for_each(cute::make_seq<NumMMAsPerChunk>{}, [&](auto mma_id_c) {
          constexpr int mma_id = decltype(mma_id_c)::value;
          constexpr int k_block = chunk_id * NumMMAsPerChunk + mma_id;

          warpgroup_arrive();
          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma, tCrA_mma(_, _, cute::Int<k_block>{}),
                     tCrB(_, _, cute::Int<k_block>{}, read_stage),
                     intermediate_array[chunk_id]);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;

          if constexpr (k_block == K_BLOCK_MAX - 1) {
            pipeline.consumer_release(
                smem_pipe_release);  // UNLOCK smem_pipe_release, done _computing_ on it
            ++smem_pipe_release;
          }

          if constexpr (k_block == 0) {
            barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            copy_scale_for_tile(read_stage);
          }

          if constexpr (k_block == K_BLOCK_MAX - 1) {
            // The last k_block

            pipeline.consumer_wait(smem_pipe_read, barrier_token);
            Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 0,
                                  smem_pipe_read.index());
            Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, 1,
                                  smem_pipe_read.index());

            warpgroup_commit_batch();
            warpgroup_wait<0>();

            warpgroup_fence_operand(intermediate_array[chunk_id]);

            scale_intermediate(intermediate_array[chunk_id], chunk_id, read_stage, false);

            Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma, cute::Int<0>{});
          } else {
            if constexpr (k_block < K_BLOCK_MAX - 2) {
              Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM,
                                    k_block + 2, read_stage);
            }
            Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma,
                                    cute::Int<k_block + 1>{});
          }
        });

        warpgroup_commit_batch();

        if constexpr (chunk_id > 0) {
          warpgroup_wait<1>();

          constexpr int chunk_id_ = chunk_id - 1;
          warpgroup_fence_operand(intermediate_array[chunk_id_]);

          scale_intermediate(intermediate_array[chunk_id_], chunk_id_, read_stage, false);
        }
      });
    }

    {
      //
      // Last k tile
      //
      Tensor intermediate = make_fragment_like(accum);

      int read_stage = smem_pipe_read.index();

      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

      // Unroll the K mode manually to set scale D to 1
      cute::for_each(cute::make_seq<K_BLOCK_MAX>{}, [&](auto k_block_c) {
        constexpr int k_block = decltype(k_block_c)::value;

        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA_mma(_, _, cute::Int<k_block>{}),
                   tCrB(_, _, cute::Int<k_block>{}, read_stage), intermediate);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;

        if constexpr (k_block == 0) {
          copy_scale_for_tile(read_stage);
        }

        if constexpr (k_block == K_BLOCK_MAX - 1) {
          // release prior barrier
          pipeline.consumer_release(
              smem_pipe_release);  // UNLOCK smem_pipe_release, done _computing_ on it
          ++smem_pipe_release;
        }

        if constexpr (k_block < K_BLOCK_MAX - 2) {
          Utils::copy_tensors_A(smem_tiled_copy_A_LDSM, tCsA_LDSM, tCrA_copy_view_LDSM, k_block + 2,
                                read_stage);
        }
        if constexpr (k_block < K_BLOCK_MAX - 1) {
          Utils::convert_A_kblock(tCrA_load_4b_packed, tCrA_mma,
                                  cute::Int<k_block + 1>{});
        }

        if constexpr ((k_block + 1) % NumMMAsPerChunk == 0) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

          warpgroup_commit_batch();
          warpgroup_wait<0>();
          warpgroup_fence_operand(intermediate);

          constexpr int scale_idx = k_block / NumMMAsPerChunk;
          scale_intermediate(intermediate, scale_idx, read_stage, false);
        }
      });
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release,
                               int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = 1;
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(
          smem_pipe_release);  // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //
  CUTLASS_DEVICE auto tensormaps_init(Params const& mainloop_params,
                                      TensorMapStorage& shared_tensormaps, int32_t sm_count,
                                      int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap =
        reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];

    // Bringing tensormaps from params to smem for modification later
    Tensor pA_tensormap =
        make_tensor(mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sA_tensormap =
        make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
    Tensor pB_tensormap =
        make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sB_tensormap =
        make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

    if (cute::elect_one_sync()) {
      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
    }

    __syncwarp();

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(tma_desc_a, tma_desc_b);
    } else if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      return cute::make_tuple(tma_desc_a, tma_desc_b);
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in tensormaps_init.");
    }
  }

  // Replace address for the global tensor (to be done by single thread)
  template <class... TMs>
  CUTLASS_DEVICE void tensormaps_replace_global_address(TensorMapStorage& shared_tensormaps,
                                                        Params const& mainloop_params,
                                                        cute::tuple<TMs...> const& input_tensormaps,
                                                        int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    mainloop_params.ptr_B[next_batch]);

    if (TensormapUpdateShapesStridesForAandScale) {
      cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                      mainloop_params.ptr_A[next_batch]);
    } else {
      cute::tma_descriptor_replace_addr_in_global_mem(get<0>(input_tensormaps),
                                                      mainloop_params.ptr_A[next_batch]);
    }
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by
  // single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE void tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps, Params const& mainloop_params, int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);

    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0, 0, 0, 0, 0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0, 0, 0, 0, 0};
    SwappedElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(
        ptr_B,
        detail::get_gmem_layout(make_shape(N, K, Int<1>{}), mainloop_params.ptr_dB[next_group]));
    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b, prob_shape_B,
                                             prob_stride_B);

    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<SwappedElementB>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B, prob_stride_B);

    if (TensormapUpdateShapesStridesForAandScale) {
      SwappedElementA const* ptr_A = nullptr;
      Tensor tensor_a = make_tensor(
          ptr_A,
          detail::get_gmem_layout(make_shape(M, K, Int<1>{}), mainloop_params.ptr_dA[next_group]));
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a, prob_shape_A,
                                               prob_stride_A);

      // Convert strides to byte strides
      for (uint64_t& stride : prob_stride_A) {
        stride = (stride * sizeof_bits_v<SwappedElementA>) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A,
                                                              prob_shape_A, prob_stride_A);
    }
  }

  template <class... TMs, class ProblemShape_MNKL>
  CUTLASS_DEVICE void tensormaps_perform_update(TensorMapStorage& shared_tensormaps,
                                                Params const& mainloop_params,
                                                cute::tuple<TMs...> const& input_tensormaps,
                                                ProblemShape_MNKL problem_shape_mnkl,
                                                int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, input_tensormaps,
                                        next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps, mainloop_params, next_batch,
                                                    problem_shape_mnkl);
      }
    }
  }

  template <class... TMs>
  CUTLASS_DEVICE void tensormaps_cp_fence_release(TensorMapStorage& shared_tensormaps,
                                                  cute::tuple<TMs...> const& input_tensormaps) {
    // [None][fix] Fix W4A8 MoE kernel issue
    // https://github.com/NVIDIA/TensorRT-LLM/pull/7072
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }

    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);

    if (TensormapUpdateShapesStridesForAandScale) {
      TensormapUpdateShapesStridesForAandScale = false;

      tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A);
    } else {
      tma_descriptor_fence_release();
    }
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class... TMs>
  CUTLASS_DEVICE void tensormaps_fence_acquire(cute::tuple<TMs...> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE InputTensors tensors_perform_update(
      InputTensors const& input_tensors, [[maybe_unused]] Params const& mainloop_params,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl, [[maybe_unused]] int32_t next_batch) {
    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      return cute::make_tuple(get<0>(input_tensors), get<1>(input_tensors),
                              mainloop_params.ptr_S[next_batch], get<3>(input_tensors),
                              get<4>(input_tensors));
    }
    return input_tensors;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
