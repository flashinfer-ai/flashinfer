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

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#include "cute/tensor.hpp"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_conversion.h"
#include "cutlass/tensor_ref.h"
#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif

#include <cuda.h>
#include <cuda_fp16.h>
#include <math.h>

#include <sstream>

#include "../include/moe_gemm_kernels.h"
#include "launchers/moe_gemm_tma_ws_mixed_input_launcher.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.h"

namespace tensorrt_llm::kernels::cutlass_kernels_oss {

#if defined(ENABLE_FP4)
using tensorrt_llm::kernels::cutlass_kernels::Fp4Type;
#endif
using tensorrt_llm::kernels::cutlass_kernels::GroupedGemmInput;
using tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;
using tensorrt_llm::kernels::cutlass_kernels::Wfp4Afp8ScaleMode;
namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

using namespace cute;

// PHASE3_SINGLE_CONFIG_TEMP: compile one selected Humming-style SM90 mixed-input
// tile/cluster while completing runtime per-token scale semantics. Restore the
// normal tile/cluster dispatch before the final PR shape.
#ifdef FLASHINFER_CUTLASS_PHASE3_SINGLE_CONFIG
#if defined(FLASHINFER_CUTLASS_PHASE3_M64N16K128_C1X1)
using Phase3SingleConfigTileShape = Shape<_64, _16, _128>;
using Phase3SingleConfigClusterShape = Shape<_1, _1, _1>;
static constexpr auto Phase3SingleConfigTile =
    tkc::CutlassTileConfigSM90::CtaShape64x16x128B;
static constexpr auto Phase3SingleConfigCluster =
    tkc::ClusterShape::ClusterShape_1x1x1;
#elif defined(FLASHINFER_CUTLASS_PHASE3_M64N32K512_C2X1)
using Phase3SingleConfigTileShape = Shape<_64, _32, _512>;
using Phase3SingleConfigClusterShape = Shape<_2, _1, _1>;
static constexpr auto Phase3SingleConfigTile =
    tkc::CutlassTileConfigSM90::CtaShape64x32x512B;
static constexpr auto Phase3SingleConfigCluster =
    tkc::ClusterShape::ClusterShape_2x1x1;
#elif defined(FLASHINFER_CUTLASS_PHASE3_M64N64K512_C1X2)
using Phase3SingleConfigTileShape = Shape<_64, _64, _512>;
using Phase3SingleConfigClusterShape = Shape<_1, _2, _1>;
static constexpr auto Phase3SingleConfigTile =
    tkc::CutlassTileConfigSM90::CtaShape64x64x512B;
static constexpr auto Phase3SingleConfigCluster =
    tkc::ClusterShape::ClusterShape_1x2x1;
#else
#error "Unknown FLASHINFER_CUTLASS_PHASE3_SINGLE_CONFIG selection"
#endif
#endif

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          typename CTAShape, typename ClusterShape,
          cutlass::gemm::collective::MixedInputScaleMode ScaleMode>
void sm90_dispatch_mainloop_schedules(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS
#ifdef FLASHINFER_CUTLASS_PHASE3_SINGLE_CONFIG
  // PHASE3_SINGLE_CONFIG_TEMP: the debug tactic set only instantiates pingpong.
  if (inputs.gemm_config.mainloop_schedule != tkc::MainloopScheduleType::PINGPONG) {
    TLLM_THROW("[Mixed dtype MoE GEMM] phase3 single-config smoke expects pingpong.");
  }
  sm90_generic_mixed_moe_gemm_kernelLauncher<
      T, WeightType, GemmOutputType, EpilogueTag, CTAShape, ClusterShape,
      cutlass::gemm::KernelTmaWarpSpecializedPingpong,
      cutlass::epilogue::TmaWarpSpecializedCooperative,
      cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, ScaleMode>(
      inputs, hopper_inputs, sm_count_, workspace_size);
#else
  switch (inputs.gemm_config.mainloop_schedule) {
    case tkc::MainloopScheduleType::COOPERATIVE:
      if constexpr (get<0>(CTAShape{}) < 128) {
        TLLM_THROW("COOPERATIVE is only enabled when tile M >= 128.");
      } else {
        sm90_generic_mixed_moe_gemm_kernelLauncher<
            T, WeightType, GemmOutputType, EpilogueTag, CTAShape, ClusterShape,
            cutlass::gemm::KernelTmaWarpSpecializedCooperative,
            cutlass::epilogue::TmaWarpSpecializedCooperative,
            cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, ScaleMode>(
            inputs, hopper_inputs, sm_count_, workspace_size);
      }

      break;
    case tkc::MainloopScheduleType::PINGPONG:
      sm90_generic_mixed_moe_gemm_kernelLauncher<
          T, WeightType, GemmOutputType, EpilogueTag, CTAShape, ClusterShape,
          cutlass::gemm::KernelTmaWarpSpecializedPingpong,
          cutlass::epilogue::TmaWarpSpecializedCooperative,
          cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, ScaleMode>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    default:
      TLLM_THROW(
          "[Mixed dtype MoE GEMM][sm90_dispatch_mainloop_schedules] mainloop schedule config is "
          "invalid "
          "for "
          "mixed type GEMM.");
      break;
  }
#endif
#else
  TLLM_THROW(
      "Please recompile with support for hopper by passing 90-real as an arch to build_wheel.py.");
#endif
}

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          typename CTAShape, cutlass::gemm::collective::MixedInputScaleMode ScaleMode>
void sm90_dispatch_moe_mixed_dtype_gemm_config(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
#ifdef FLASHINFER_CUTLASS_PHASE3_SINGLE_CONFIG
  // PHASE3_SINGLE_CONFIG_TEMP: route only the selected debug cluster shape.
  if (inputs.gemm_config.cluster_shape != Phase3SingleConfigCluster) {
    TLLM_THROW("[Mixed dtype MoE GEMM] phase3 single-config received unexpected cluster.");
  }
  sm90_dispatch_mainloop_schedules<T, WeightType, GemmOutputType, EpilogueTag, CTAShape,
                                   Phase3SingleConfigClusterShape, ScaleMode>(
      inputs, hopper_inputs, sm_count_, workspace_size);
#else
  switch (inputs.gemm_config.cluster_shape) {
    case tkc::ClusterShape::ClusterShape_1x1x1:
      sm90_dispatch_mainloop_schedules<T, WeightType, GemmOutputType, EpilogueTag, CTAShape,
                                       Shape<_1, _1, _1>, ScaleMode>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::ClusterShape::ClusterShape_2x1x1:
      sm90_dispatch_mainloop_schedules<T, WeightType, GemmOutputType, EpilogueTag, CTAShape,
                                       Shape<_2, _1, _1>, ScaleMode>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::ClusterShape::ClusterShape_1x2x1:
      sm90_dispatch_mainloop_schedules<T, WeightType, GemmOutputType, EpilogueTag, CTAShape,
                                       Shape<_1, _2, _1>, ScaleMode>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
      sm90_dispatch_mainloop_schedules<T, WeightType, GemmOutputType, EpilogueTag, CTAShape,
                                       Shape<_2, _2, _1>, ScaleMode>(
          inputs, hopper_inputs, sm_count_, workspace_size);
      break;
    default:
      TLLM_THROW(
          "[Mixed dtype MoE GEMM][dispatch_CGA_config] Config is invalid for mixed type GEMM.");
      break;
  }
#endif
}

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag,
          int UnusedScalePackFactor,
          Wfp4Afp8ScaleMode Wfp4Afp8Mode = Wfp4Afp8ScaleMode::kNone>
void sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass(
    GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> inputs,
    TmaWarpSpecializedGroupedGemmInput hopper_inputs, int sm_count_, size_t* workspace_size) {
  TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
  // We also only instantiate configs here where threadblockShapeM == warpShapeM since those usually
  // perform the best for mixed type gemms.

#if defined(ENABLE_FP4)
  TLLM_CHECK(sizeof(T) == (std::is_same_v<WeightType, Fp4Type>) ? 2 : 1);
#else
  TLLM_CHECK(sizeof(T) == 1);
#endif
  static_cast<void>(UnusedScalePackFactor);
  static constexpr auto ScaleMode =
      Wfp4Afp8Mode == Wfp4Afp8ScaleMode::kHummingPreMmaE8M0
          ? cutlass::gemm::collective::MixedInputScaleMode::kPreMmaE8M0
          : cutlass::gemm::collective::MixedInputScaleMode::kPostMma;

#define DISPATCH_MIXED_DTYPE_MOE_TILE(ENUM_NAME, TILE_M, TILE_N, TILE_K)                    \
  case tkc::CutlassTileConfigSM90::ENUM_NAME:                                                \
    sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag,    \
                                              Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>,  \
                                              ScaleMode>(                                    \
        inputs, hopper_inputs, sm_count_, workspace_size);                                   \
    break

#ifdef FLASHINFER_CUTLASS_PHASE3_SINGLE_CONFIG
  // PHASE3_SINGLE_CONFIG_TEMP: route only the selected debug CTA tile.
  if (inputs.gemm_config.tile_config_sm90 != Phase3SingleConfigTile) {
    TLLM_THROW("[Mixed dtype MoE GEMM] phase3 single-config received unexpected CTA tile.");
  }
  sm90_dispatch_moe_mixed_dtype_gemm_config<T, WeightType, GemmOutputType, EpilogueTag,
                                            Phase3SingleConfigTileShape, ScaleMode>(
      inputs, hopper_inputs, sm_count_, workspace_size);
#else
  switch (inputs.gemm_config.tile_config_sm90) {
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x16x128B, 64, 16, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x16x256B, 64, 16, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x16x512B, 64, 16, 512);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x32x128B, 64, 32, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x32x256B, 64, 32, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x32x512B, 64, 32, 512);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x64x128B, 64, 64, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x64x256B, 64, 64, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x64x512B, 64, 64, 512);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x128x128B, 64, 128, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x128x256B, 64, 128, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape64x128x512B, 64, 128, 512);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x16x128B, 128, 16, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x16x256B, 128, 16, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x16x512B, 128, 16, 512);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x32x128B, 128, 32, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x32x256B, 128, 32, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x32x512B, 128, 32, 512);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x64x128B, 128, 64, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x64x256B, 128, 64, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x64x512B, 128, 64, 512);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x128x128B, 128, 128, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x128x256B, 128, 128, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x128x512B, 128, 128, 512);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x256x128B, 128, 256, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape128x256x256B, 128, 256, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape256x128x128B, 256, 128, 128);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape256x128x256B, 256, 128, 256);
    DISPATCH_MIXED_DTYPE_MOE_TILE(CtaShape256x256x128B, 256, 256, 128);
    case tkc::CutlassTileConfigSM90::Undefined:
      TLLM_THROW(
          "[Mixed dtype MoE GEMM][sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass] gemm config "
          "undefined.");
      break;
    case tkc::CutlassTileConfigSM90::ChooseWithHeuristic:
      TLLM_THROW(
          "[Mixed dtype MoE GEMM][sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass] gemm config "
          "should have already "
          "been set by "
          "heuristic.");
      break;
    default:
      TLLM_THROW(
          "[Mixed dtype MoE GEMM][sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass] Config is invalid "
          "for mixed type "
          "GEMM.");
      break;
  }
#endif

#undef DISPATCH_MIXED_DTYPE_MOE_TILE
}

template <typename T, typename WeightType, typename OutputType,
          Wfp4Afp8ScaleMode Wfp4Afp8Mode = Wfp4Afp8ScaleMode::kNone>
size_t calcMaxWorkspaceSizeTmaWarpSpecializedMixedInput(int num_experts, int sm_count_) {
  size_t count = 0;
#if defined(ENABLE_FP4)
  constexpr int Ktile = (std::is_same_v<WeightType, Fp4Type>) ? 256 : 512;
#else
  constexpr int Ktile = 512;
#endif
  using _Ktile = Int<Ktile>;
  static constexpr auto ScaleMode =
      Wfp4Afp8Mode == Wfp4Afp8ScaleMode::kHummingPreMmaE8M0
          ? cutlass::gemm::collective::MixedInputScaleMode::kPreMmaE8M0
          : cutlass::gemm::collective::MixedInputScaleMode::kPostMma;

#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS
  GroupedGemmInput<T, WeightType, OutputType, OutputType> inputs{};
  inputs.num_experts = num_experts;
#ifdef FLASHINFER_CUTLASS_PHASE3_SINGLE_CONFIG
  // PHASE3_SINGLE_CONFIG_TEMP: workspace query for the selected debug tactic.
  sm90_generic_mixed_moe_gemm_kernelLauncher<T, WeightType, OutputType,
                                             tensorrt_llm::cutlass_extensions::EpilogueOpDefault,
                                             Phase3SingleConfigTileShape,
                                             Phase3SingleConfigClusterShape,
                                             cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                             cutlass::epilogue::TmaWarpSpecializedCooperative,
                                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
                                             ScaleMode>(
      inputs, TmaWarpSpecializedGroupedGemmInput{}, sm_count_, &count);
#else
  sm90_generic_mixed_moe_gemm_kernelLauncher<T, WeightType, OutputType,
                                             tensorrt_llm::cutlass_extensions::EpilogueOpDefault,
                                             Shape<_128, _64, _Ktile>, Shape<_1, _1, _1>,
                                             cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                                             cutlass::epilogue::TmaWarpSpecializedCooperative,
                                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY,
                                             ScaleMode>(
      inputs, TmaWarpSpecializedGroupedGemmInput{}, sm_count_, &count);
#endif
#endif
  return count;
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels_oss
