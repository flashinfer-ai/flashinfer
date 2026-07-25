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

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__

#include <cstdint>

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue/collective/default_epilogue_array_per_token_scale.hpp"
#include "cutlass_extensions/epilogue/collective/sm90_epilogue_array_tma_warpspecialized_mixed_input.hpp"
#include "cutlass_extensions/epilogue/fusion/sm90_ptr_array_per_token_scale_callbacks_tma_warpspecialized.hpp"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_single_warpgroup_persistent.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative_precomputed.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_pingpong_precomputed.hpp"
#include "cutlass_extensions/gemm_configs.h"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif  // __GNUC__

#include "moe_gemm_tma_ws_mixed_input_launcher.h"
#include "moe_gemm_tma_ws_mixed_input_prebuild.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels_oss {
using namespace tensorrt_llm::kernels::cutlass_kernels;
#ifdef ENABLE_FP4
using SafeFP4 = Fp4Type;
#else
struct SafeFP4 {};
#endif
namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

using namespace cute;

namespace mixed_input_detail {

template <bool UseSingleWarpgroup, class TileShape, class ClusterShape, class ElementAccumulator,
          class ElementC, class LayoutC, int AlignmentC, class ElementD, class LayoutD,
          int AlignmentD, class EpilogueSchedule, class FusionOperation>
struct EpilogueSelector;

template <class TileShape, class ClusterShape, class ElementAccumulator, class ElementC,
          class LayoutC, int AlignmentC, class ElementD, class LayoutD, int AlignmentD,
          class EpilogueSchedule, class FusionOperation>
struct EpilogueSelector<false, TileShape, ClusterShape, ElementAccumulator, ElementC, LayoutC,
                        AlignmentC, ElementD, LayoutD, AlignmentD, EpilogueSchedule,
                        FusionOperation> {
  using Type = typename tensorrt_llm::cutlass_extensions::epilogue::collective::
      MixedInputSm90TmaEpilogueBuilder<
          cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
          ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type*, AlignmentC, ElementD,
          typename cutlass::layout::LayoutTranspose<LayoutD>::type*, AlignmentD, EpilogueSchedule,
          FusionOperation>::CollectiveOp;
};

template <class TileShape, class ClusterShape, class ElementAccumulator, class ElementC,
          class LayoutC, int AlignmentC, class ElementD, class LayoutD, int AlignmentD,
          class EpilogueSchedule, class FusionOperation>
struct EpilogueSelector<true, TileShape, ClusterShape, ElementAccumulator, ElementC, LayoutC,
                        AlignmentC, ElementD, LayoutD, AlignmentD, EpilogueSchedule,
                        FusionOperation> {
  using EpilogueLayoutC = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
  using EpilogueLayoutD = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  using Epilogue = cutlass::epilogue::collective::SmemEpilogueArrayPerTokenScale<
      TileShape, ElementC, cutlass::detail::TagToStrideC_t<EpilogueLayoutC*>, ElementD,
      cutlass::detail::TagToStrideC_t<EpilogueLayoutD*>, ElementAccumulator, ElementAccumulator>;
  using Type = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<Epilogue>;
};

template <bool UseSingleWarpgroup, class ProblemShape, class CollectiveMainloop,
          class CollectiveEpilogue, int CtasPerSm, bool RollingRefill>
struct GemmKernelSelector;

template <class ProblemShape, class CollectiveMainloop, class CollectiveEpilogue, int CtasPerSm,
          bool RollingRefill>
struct GemmKernelSelector<false, ProblemShape, CollectiveMainloop, CollectiveEpilogue, CtasPerSm,
                          RollingRefill> {
  using Type =
      cutlass::gemm::kernel::GemmUniversalPrecomputedScheduler<ProblemShape, CollectiveMainloop,
                                                               CollectiveEpilogue>;
};

template <class ProblemShape, class CollectiveMainloop, class CollectiveEpilogue, int CtasPerSm,
          bool RollingRefill>
struct GemmKernelSelector<true, ProblemShape, CollectiveMainloop, CollectiveEpilogue, CtasPerSm,
                          RollingRefill> {
  using Type = cutlass::gemm::kernel::SingleWarpgroupPersistentGemm<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, CtasPerSm, 3,
      RollingRefill ? cutlass::gemm::kernel::SingleWarpgroupPipelineMode::RollingRefill
                    : cutlass::gemm::kernel::SingleWarpgroupPipelineMode::PrefillAll>;
};

}  // namespace mixed_input_detail

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          typename CTAShape, typename ClusterShape, typename MainloopScheduleType,
          typename EpilogueScheduleType, cutlass::WeightOnlyQuantOp QuantOp,
          cutlass::gemm::collective::MixedInputScaleMode ScaleMode,
          tkc::MainloopScheduleType KernelType>
void sm90_generic_mixed_moe_gemm_kernelLauncher_impl(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

  /////////////////////////////////////////////////////////////////////////////////////////////////
  /// GEMM kernel configurations
  /////////////////////////////////////////////////////////////////////////////////////////////////

  // A matrix configuration
  using ElementA = typename TllmToCutlassTypeAdapter<T>::type;
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Alignment of A matrix in units of elements
                                                    // (up to 16 bytes)

  // B matrix configuration
  using ElementB_ = typename TllmToCutlassTypeAdapter<WeightType>::type;
  using ElementB = std::conditional_t<std::is_same_v<WeightType, cutlass::uint4b_t>,
                                      cutlass::int4b_t, ElementB_>;
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                    // matrix in units of elements (up to 16 bytes)

  // This example manually swaps and transposes, so keep transpose of input layouts
  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

  // Need to pass a pointer type to make the 3rd dimension of Stride be _0
  using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

  // Scale configuration
  constexpr bool use_mxfp4_weight = std::is_same_v<ElementB, cutlass::float_e2m1_t>;
  constexpr int group_size = use_mxfp4_weight ? cutlass::gemm::collective::detail::mxfp4_group_size
                                              : cutlass::gemm::collective::detail::int4_group_size;
  using ElementScale =
      std::conditional_t<use_mxfp4_weight, cutlass::float_ue8m0_t,
                         TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA>;
  using LayoutScale = cutlass::layout::RowMajor;

  // C/D matrix configuration
  using ElementC = typename TllmToCutlassTypeAdapter<GemmOutputType>::type;
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of C
                                                    // matrix in units of elements (up to 16 bytes)

  // D matrix configuration
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // Core kernel configurations
  using ElementAccumulator = float;  // Element type for internal accumulation
  using ArchTag =
      cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using TileShape = CTAShape;                            // Threadblock-level tile size
  using KernelSchedule = std::conditional_t<
      std::is_same_v<MainloopScheduleType, cutlass::gemm::KernelTmaWarpSpecializedPingpong>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
  using EpilogueSchedule = std::conditional_t<
      std::is_same_v<KernelSchedule, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;  // Epilogue to launch
  constexpr bool use_fused_e8m0_scale =
      ScaleMode == cutlass::gemm::collective::MixedInputScaleMode::kPreMmaE8M0;
  constexpr bool use_single_warpgroup =
      KernelType == tkc::MainloopScheduleType::SINGLE_WARPGROUP_PREFILL ||
      KernelType == tkc::MainloopScheduleType::SINGLE_WARPGROUP_ROLLING;
  constexpr bool use_rolling_refill =
      KernelType == tkc::MainloopScheduleType::SINGLE_WARPGROUP_ROLLING;
  constexpr int SmallKTileN = cute::size<1>(TileShape{});
  constexpr int SmallKCtasPerSm = SmallKTileN <= 16 ? 5 : (SmallKTileN == 32 ? 4 : 3);

  static_assert(!use_single_warpgroup || use_fused_e8m0_scale,
                "The single-warpgroup kernel is only valid for pre-MMA E8M0 scaling.");
  static_assert(!use_single_warpgroup || cute::size(ClusterShape{}) == 1,
                "The single-warpgroup kernel requires a 1x1x1 cluster.");
  static_assert(
      !use_single_warpgroup ||
          (cute::size<0>(TileShape{}) == 128 && cute::size<2>(TileShape{}) == 128 &&
           (SmallKTileN == 8 || SmallKTileN == 16 || SmallKTileN == 32 || SmallKTileN == 40)),
      "Unsupported single-warpgroup tile shape.");

  using FusionOperation =
      std::conditional_t<use_fused_e8m0_scale,
                         cutlass::epilogue::fusion::PtrArrayPerTokenScaledAcc<
                             ElementD, ElementAccumulator, ElementAccumulator>,
                         cutlass::epilogue::fusion::LinearCombination<
                             ElementD, ElementAccumulator, ElementC, ElementAccumulator>>;

  using CollectiveEpilogue = typename mixed_input_detail::EpilogueSelector<
      use_single_warpgroup, TileShape, ClusterShape, ElementAccumulator, ElementC, LayoutC,
      AlignmentC, ElementD, LayoutD, AlignmentD, EpilogueSchedule, FusionOperation>::Type;

  // =========================================================== MIXED INPUT WITH SCALES
  // =========================================================================== The Scale
  // information must get paired with the operand that will be scaled. In this example, B is scaled
  // so we make a tuple of B's information and the scale information.
  using StageCountType =
      std::conditional_t<use_single_warpgroup, cutlass::gemm::collective::StageCount<3>,
                         cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                             sizeof(typename CollectiveEpilogue::SharedStorage))>>;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag, OperatorClass, cute::tuple<ElementB, ElementScale>, LayoutB_Transpose*, AlignmentB,
      ElementA, LayoutA_Transpose*, AlignmentA, ElementAccumulator, TileShape, ClusterShape,
      StageCountType, KernelSchedule, ScaleMode>::CollectiveOp;

  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using GemmKernel =
      typename mixed_input_detail::GemmKernelSelector<use_single_warpgroup, ProblemShape,
                                                      CollectiveMainloop, CollectiveEpilogue,
                                                      SmallKCtasPerSm, use_rolling_refill>::Type;

  using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
  using StrideS = typename CollectiveMainloop::StrideScale;

  GemmGrouped gemm;
  using Args = typename GemmGrouped::Arguments;
  Args arguments;

  decltype(arguments.epilogue.thread) fusion_args;
  if constexpr (use_fused_e8m0_scale) {
    fusion_args.token_scale_default = ElementAccumulator(1);
    fusion_args.token_scale_ptr_array = inputs.alpha_scales;
  } else {
    fusion_args.alpha = use_mxfp4_weight ? 1 : 0;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = use_mxfp4_weight ? nullptr : inputs.alpha_scales;
    fusion_args.beta_ptr_array = nullptr;
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, use_mxfp4_weight ? 0 : 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, use_mxfp4_weight ? 0 : 1};
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = use_single_warpgroup ? sm_count_ * SmallKCtasPerSm : sm_count_;

  if constexpr (use_single_warpgroup) {
    arguments = Args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {inputs.num_experts, hopper_inputs.int4_groupwise_params.shape.problem_shapes, nullptr},
        {reinterpret_cast<ElementB const**>(hopper_inputs.ptr_weight),
         reinterpret_cast<StrideB*>(hopper_inputs.stride_weight),
         reinterpret_cast<ElementA const**>(hopper_inputs.ptr_act),
         reinterpret_cast<StrideA*>(hopper_inputs.stride_act),
         reinterpret_cast<ElementScale const**>(hopper_inputs.int4_groupwise_params.ptr_s_a),
         reinterpret_cast<StrideS*>(hopper_inputs.int4_groupwise_params.stride_s_a), group_size},
        {fusion_args, nullptr, reinterpret_cast<StrideC*>(hopper_inputs.stride_c),
         reinterpret_cast<ElementD**>(hopper_inputs.ptr_d),
         reinterpret_cast<StrideD*>(hopper_inputs.stride_d), reinterpret_cast<ElementD*>(inputs.C),
         inputs.n, inputs.n, ElementAccumulator(0)},
        hw_info};
  } else {
    arguments = Args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {inputs.num_experts, hopper_inputs.int4_groupwise_params.shape.problem_shapes, nullptr},
        {reinterpret_cast<ElementB const**>(hopper_inputs.ptr_weight),
         reinterpret_cast<StrideB*>(hopper_inputs.stride_weight),
         reinterpret_cast<ElementA const**>(hopper_inputs.ptr_act),
         reinterpret_cast<StrideA*>(hopper_inputs.stride_act),
         reinterpret_cast<ElementScale const**>(hopper_inputs.int4_groupwise_params.ptr_s_a),
         reinterpret_cast<StrideS*>(hopper_inputs.int4_groupwise_params.stride_s_a), group_size},
        {fusion_args, reinterpret_cast<ElementC const**>(hopper_inputs.ptr_c),
         reinterpret_cast<StrideC*>(hopper_inputs.stride_c),
         reinterpret_cast<ElementD**>(hopper_inputs.ptr_d),
         reinterpret_cast<StrideD*>(hopper_inputs.stride_d)},
        hw_info};
  }

  // Optimize tile scheduling for better L2 locality
  using RasterOrderOptions =
      typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;
  arguments.scheduler.max_swizzle_size = detail::kPrecomputedSchedulerMaxSwizzle;
  arguments.scheduler.raster_order = RasterOrderOptions::AlongM;

  assert(group_size == int(inputs.groupwise_quant_group_size));
  if (workspace_size != nullptr) {
    *workspace_size = gemm.get_workspace_size(arguments);
    return;
  }

  if constexpr (use_single_warpgroup) {
    TLLM_CHECK_WITH_INFO(inputs.k > 0 && inputs.k % 128 == 0,
                         "Single-warpgroup GEMM requires K to be a positive multiple of 128.");
    if constexpr (use_rolling_refill) {
      TLLM_CHECK_WITH_INFO(inputs.k > 384,
                           "Rolling-refill single-warpgroup GEMM requires K > 384.");
    } else {
      TLLM_CHECK_WITH_INFO(inputs.k <= 384, "Prefill single-warpgroup GEMM requires K <= 384.");
    }
    TLLM_CHECK_WITH_INFO(inputs.n > 0 && inputs.n % 128 == 0,
                         "Single-warpgroup GEMM requires output channels to be 128 aligned.");
    TLLM_CHECK_WITH_INFO(
        inputs.C != nullptr && reinterpret_cast<std::uintptr_t>(inputs.C) % 16 == 0,
        "Single-warpgroup GEMM requires a 16B-aligned contiguous output base.");
  }

  static constexpr int CurrentTileShapeM = cute::size<0>(TileShape{});
  static constexpr int CurrentTileShapeN = cute::size<1>(TileShape{});
  static constexpr int CurrentClusterShapeM = cute::size<0>(ClusterShape{});
  static constexpr int CurrentClusterShapeN = cute::size<1>(ClusterShape{});
  int64_t const total_routed_tokens = hopper_inputs.precomputed_scheduler_total_routed_tokens;
  TLLM_CHECK_WITH_INFO(total_routed_tokens >= 0,
                       "Precomputed scheduler requires a nonnegative routed token count.");
  if (total_routed_tokens == 0) {
    return;
  }
  auto precomputed_workspace =
      detail::partition_precomputed_scheduler_workspace<CurrentTileShapeM, CurrentTileShapeN,
                                                        CurrentClusterShapeM, CurrentClusterShapeN,
                                                        use_single_warpgroup>(
          hopper_inputs, inputs.num_experts, total_routed_tokens, inputs.n, hw_info.sm_count);
  arguments.scheduler.precomputed_work_tiles = precomputed_workspace.work_tiles;
  if constexpr (use_single_warpgroup) {
    arguments.scheduler.precomputed_work_tiles_per_worker =
        precomputed_workspace.work_tiles_per_worker;
  }
  arguments.mainloop.ptr_A_prebuilt_tma_desc = precomputed_workspace.prebuilt_tma_desc_A;
  arguments.mainloop.ptr_B_prebuilt_tma_descs = precomputed_workspace.prebuilt_tma_desc_B;

  if (gemm.get_workspace_size(arguments) > hopper_inputs.gemm_workspace_size) {
    TLLM_LOG_ERROR("[Mixed dtype WS grouped GEMM] given workspace size insufficient, %d < %d.",
                   gemm.get_workspace_size(arguments), hopper_inputs.gemm_workspace_size);
  }

  // This is not initialized during workspace size calculation so check after
  TLLM_CHECK_WITH_INFO(hopper_inputs.swap_ab,
                       "swap_ab must be true for mixed dtype WS grouped GEMM");

  auto can_implement = gemm.can_implement(arguments);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "mixed dtype WS grouped cutlass kernel will fail for params. Error: " +
                          std::string(cutlass::cutlassGetStatusString(can_implement));
    std::cout << err_msg << std::endl;
    throw std::runtime_error("[Mixed dtype WS grouped GEMM] " + err_msg);
  }

  auto init_status = gemm.initialize(arguments, hopper_inputs.gemm_workspace, inputs.stream);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to initialize cutlass mixed dtype WS grouped gemm. Error: " +
                          std::string(cutlass::cutlassGetStatusString(init_status));
    throw std::runtime_error("[Mixed dtype WS grouped GEMM] " + err_msg);
  }

  detail::build_precomputed_work_tile_map<CurrentTileShapeM, CurrentTileShapeN,
                                          CurrentClusterShapeM, CurrentClusterShapeN,
                                          use_single_warpgroup>(
      precomputed_workspace, hopper_inputs.int4_groupwise_params.shape.problem_shapes,
      inputs.num_experts, total_routed_tokens, inputs.n, gemm.params().mainloop, inputs.stream);

  auto run_status = gemm.run(inputs.stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run cutlass mixed dtype WS grouped gemm. Error: " +
                          std::string(cutlass::cutlassGetStatusString(run_status));
    throw std::runtime_error("[Mixed dtype WS grouped GEMM] " + err_msg);
  }
  return;
}

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          typename CTAShape, typename ClusterShape, typename MainloopScheduleType,
          typename EpilogueScheduleType, cutlass::WeightOnlyQuantOp QuantOp,
          cutlass::gemm::collective::MixedInputScaleMode ScaleMode>
void sm90_generic_mixed_moe_gemm_kernelLauncher(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  sm90_generic_mixed_moe_gemm_kernelLauncher_impl<
      T, WeightType, GemmOutputType, EpilogueTag, CTAShape, ClusterShape, MainloopScheduleType,
      EpilogueScheduleType, QuantOp, ScaleMode, tkc::MainloopScheduleType::AUTO>(
      inputs, hopper_inputs, sm_count_, workspace_size);
}

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          typename CTAShape, typename ClusterShape, typename MainloopScheduleType,
          typename EpilogueScheduleType, cutlass::WeightOnlyQuantOp QuantOp,
          cutlass::gemm::collective::MixedInputScaleMode ScaleMode,
          tkc::MainloopScheduleType KernelType>
void sm90_generic_mixed_moe_small_k_kernelLauncher(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  static_assert(KernelType == tkc::MainloopScheduleType::SINGLE_WARPGROUP_PREFILL ||
                    KernelType == tkc::MainloopScheduleType::SINGLE_WARPGROUP_ROLLING,
                "Small-K launcher requires a single-warpgroup schedule.");
  sm90_generic_mixed_moe_gemm_kernelLauncher_impl<
      T, WeightType, GemmOutputType, EpilogueTag, CTAShape, ClusterShape, MainloopScheduleType,
      EpilogueScheduleType, QuantOp, ScaleMode, KernelType>(inputs, hopper_inputs, sm_count_,
                                                            workspace_size);
}

}  // namespace cutlass_kernels_oss
}  // namespace kernels
}  // namespace tensorrt_llm
