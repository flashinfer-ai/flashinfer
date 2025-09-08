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
#ifndef FLASHINFER_FP8_GEMM_TEMPLATE_SM100_H_
#define FLASHINFER_FP8_GEMM_TEMPLATE_SM100_H_

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_conversion.h"
#include "flashinfer/arch_condition.h"
#include "flashinfer/cutlass_utils.cuh"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif  // __GNUC__

#include <cstddef>
#include <stdexcept>

#include "cutlass/float8.h"
#include "flashinfer/gemm/cutlass_gemm_configs.h"

namespace flashinfer {
namespace gemm {

template <typename T>
struct SMTypeAdapter {};

struct _1SM;
struct _2SM;

template <>
struct SMTypeAdapter<_1SM> {
  static int const Scale = 1;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
};

template <>
struct SMTypeAdapter<_2SM> {
  static int const Scale = 2;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
  using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
};

// only support fp8 TN gemm
// batch GEMM
// return require_workspace_size
template <typename T, typename arch, int32_t CTA_M_, int32_t CTA_N_, int32_t CTA_K_,
          typename ClusterShape_, typename XSM_>
size_t genericFp8GemmKernelLauncherSm100(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B,
                                         float const* scale_a, float const* scale_b, T* D, int m,
                                         int n, int k, int b, CutlassGemmConfig config,
                                         char* workspacePtr, size_t const workspaceBytes,
                                         cudaStream_t stream) {
  using namespace cute;

  // A matrix configuration
  using ElementA = cutlass::float_e4m3_t;     // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::float_e4m3_t;        // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementOutput_ =
      typename cutlass::platform::conditional<cutlass::platform::is_same<T, half>::value,
                                              cutlass::half_t, T>::type;
#ifdef ENABLE_BF16
  using ElementOutput = typename cutlass::platform::conditional<
      cutlass::platform::is_same<ElementOutput_, __nv_bfloat16>::value, cutlass::bfloat16_t,
      ElementOutput_>::type;
#else
  using ElementOutput = ElementOutput_;
#endif

  // C matrix configuration
  using ElementC = ElementOutput;                // Element type for C and D matrix operands
  using LayoutC = cutlass::layout::ColumnMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of C
                                                    // matrix in units of elements (up to 16 bytes)

  // D matrix configuration
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  // / Core kernel configurations
  using ElementAccumulator = float;  // Element type for internal accumulation
  using ElementCompute = float;      // Element type for epilogue computation
  using ArchTag =
      cutlass::arch::Sm100;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using TileShape = cute::Shape<cute::Int<CTA_M_ * SMTypeAdapter<XSM_>::Scale>, cute::Int<CTA_N_>,
                                cute::Int<CTA_K_>>;  // Threadblock-level tile size

  using ClusterShape = ClusterShape_;  // Shape of the threadblocks in a cluster
  using EpilogueSchedule = typename SMTypeAdapter<XSM_>::EpilogueSchedule;
  using MainloopSchedule = typename SMTypeAdapter<XSM_>::MainloopSchedule;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

  using CustomEVT = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::multiplies, ElementD, float,
          cutlass::FloatRoundStyle::round_to_nearest>,        // scale_a * scale_b * acc
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>,  // scale_a
      cutlass::epilogue::fusion::Sm90EVT<
          cutlass::epilogue::fusion::Sm90Compute<
              cutlass::multiplies, float, float,
              cutlass::FloatRoundStyle::round_to_nearest>,        // scale_b * acc
          cutlass::epilogue::fusion::Sm90ScalarBroadcast<float>,  // scale_b
          cutlass::epilogue::fusion::Sm90AccFetch                 // acc
          >>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
      ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      EpilogueSchedule, CustomEVT>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,  // Indicates ProblemShape
                                           CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Extract information from Gemm kernel.
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
  using ElementScalar = typename EpilogueOutputOp::ElementScalar;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutScalar = cutlass::layout::PackedVectorLayout;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, b));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, b));
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, b));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, b));

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                     {m, n, k, b},
                                     {reinterpret_cast<cutlass::float_e4m3_t const*>(A), stride_A,
                                      reinterpret_cast<cutlass::float_e4m3_t const*>(B), stride_B},
                                     {{},  // epilogue.thread
                                      nullptr,
                                      stride_C,
                                      reinterpret_cast<ElementOutput*>(D),
                                      stride_D}};

  arguments.epilogue.thread = {
      {{0.F}, {scale_a}},  // scale_a
      {
          {{0.F}, {scale_b}},  // scale_b
          {},                  // acc
          {}                   // multiplies
      },
      {}  // multiplies
  };

  Gemm gemm;

  // Return workspace size
  if (!A && !B && !D) {
    return Gemm::get_workspace_size(arguments);
  }

  if (Gemm::get_workspace_size(arguments) > workspaceBytes) {
    throw std::runtime_error("[Fp8 Gemm Runner] failed to get workspace size");
  }

  auto can_implement = gemm.can_implement(arguments);
  if (can_implement != cutlass::Status::kSuccess) {
    throw std::runtime_error("[Fp8 Gemm Runner] cutlass kernel not implemented given the params");
  }

  auto initStatus = gemm.initialize(arguments, workspacePtr);
  if (initStatus != cutlass::Status::kSuccess) {
    throw std::runtime_error("[Fp8 Gemm Runner] failed to initialize");
  }

  auto runStatus = gemm.run(stream, nullptr, true);
  if (runStatus != cutlass::Status::kSuccess) {
    throw std::runtime_error("[Fp8 Gemm Runner] failed to run");
  }
  return gemm.get_workspace_size(arguments);
}

}  // namespace gemm
}  // namespace flashinfer

#define INSTANCE_FP8_GEMM_TEMPLATE_SM100(RET_TYPE, TILE_M, TILE_N, TILE_K, CGA_M_, CGA_N_, CGA_K_, \
                                         SM_TYPE)                                                  \
  template size_t genericFp8GemmKernelLauncherSm100<                                               \
      RET_TYPE, cutlass::arch::Sm100, TILE_M, TILE_N, TILE_K,                                      \
      cute::Shape<cute::Int<CGA_M_>, cute::Int<CGA_N_>, cute::Int<CGA_K_>>, SM_TYPE>(              \
      __nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float const* scale_a, float const* scale_b,  \
      RET_TYPE* D, int m, int n, int k, int b, CutlassGemmConfig config, char* workspacePtr,       \
      size_t const workspaceBytes, cudaStream_t stream);

#endif  // FLASHINFER_FP8_GEMM_TEMPLATE_SM100_H_
