/*
 * Copyright (c) 2026 by FlashInfer team.
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
#ifndef FLASHINFER_FP8_GEMM_TEMPLATE_SM120_H_
#define FLASHINFER_FP8_GEMM_TEMPLATE_SM120_H_

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "flashinfer/cutlass_utils.cuh"
#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif  // __GNUC__

#include <cstddef>
#include <stdexcept>
#include <string>

#include "cutlass/float8.h"
#include "flashinfer/gemm/cutlass_gemm_configs.h"

namespace flashinfer {
namespace gemm {

// Per-tensor scaled FP8 TN GEMM for SM120/SM121: D = scale_a * scale_b * (A . B^T).
// A: [b, m, k] row-major, B: [b, n, k] row-major (i.e. the column-major K x N operand), D: [b, m,
// n] row-major. ClusterShape is always 1x1x1 (no multicast on this arch).
//
// The schedule follows the tile: the Cooperative kernel needs CTA_M >= 128, so smaller tiles use
// Pingpong. Both use the persistent data-parallel tile scheduler. Stream-K is intentionally not
// offered: on RTX 5090 it was within 1% of the best data-parallel tile on the shapes we measured,
// and it would make the tuned and untuned reduction orders differ.
// Returns the workspace size the kernel needs; launches nothing when A, B and D are all null.
template <typename T, int32_t CTA_M_, int32_t CTA_N_, int32_t CTA_K_>
size_t genericFp8GemmKernelLauncherSm120(__nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B,
                                         float const* scale_a, float const* scale_b, T* D, int m,
                                         int n, int k, int b, char* workspacePtr,
                                         size_t const workspaceBytes, cudaStream_t stream) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  using namespace cute;

  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = cutlass_dtype_t<T>;
  using LayoutD = cutlass::layout::RowMajor;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using TileShape = Shape<Int<CTA_M_>, Int<CTA_N_>, Int<CTA_K_>>;
  using ClusterShape = Shape<_1, _1, _1>;

  using MainloopSchedule =
      std::conditional_t<(CTA_M_ >= 128), cutlass::gemm::KernelTmaWarpSpecializedCooperative,
                         cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
  // The auto epilogue tile does not divide the small-M tiles; give those an explicit one.
  using EpilogueTileType = std::conditional_t<(CTA_M_ < 64), Shape<Int<CTA_M_>, _32>,
                                              cutlass::epilogue::collective::EpilogueTileAuto>;

  using CustomEVT = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::multiplies, ElementD, ElementCompute,
          cutlass::FloatRoundStyle::round_to_nearest>,                 // a*(b*acc)
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementCompute>,  // scale_a
      cutlass::epilogue::fusion::Sm90EVT<
          cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementCompute,
                                                 ElementCompute,
                                                 cutlass::FloatRoundStyle::round_to_nearest>,
          cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementCompute>,  // scale_b
          cutlass::epilogue::fusion::Sm90AccFetch>>;                       // acc

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType, ElementAccumulator,
      ElementCompute, void, LayoutD, AlignmentD, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto, CustomEVT>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, b));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, b));
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, b));

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                     {m, n, k, b},
                                     {reinterpret_cast<ElementA const*>(A), stride_A,
                                      reinterpret_cast<ElementB const*>(B), stride_B},
                                     {{},  // epilogue.thread
                                      nullptr,
                                      stride_D,
                                      reinterpret_cast<ElementD*>(D),
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
    throw std::runtime_error("[Fp8 Gemm Runner SM120] workspace too small");
  }

  auto can_implement = gemm.can_implement(arguments);
  if (can_implement != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("[Fp8 Gemm Runner SM120] cutlass kernel cannot run the "
                                         "given problem: ") +
                             cutlass::cutlassGetStatusString(can_implement));
  }

  auto initStatus = gemm.initialize(arguments, workspacePtr, stream);
  if (initStatus != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("[Fp8 Gemm Runner SM120] failed to initialize: ") +
                             cutlass::cutlassGetStatusString(initStatus));
  }

  auto runStatus = gemm.run(arguments, workspacePtr, stream, nullptr, /*launch_with_pdl=*/true);
  if (runStatus != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("[Fp8 Gemm Runner SM120] failed to run: ") +
                             cutlass::cutlassGetStatusString(runStatus));
  }
  return Gemm::get_workspace_size(arguments);
#else
  throw std::runtime_error("[Fp8 Gemm Runner SM120] not compiled with SM120/SM121 support");
#endif
}

}  // namespace gemm
}  // namespace flashinfer

#define INSTANCE_FP8_GEMM_TEMPLATE_SM120(RET_TYPE, TILE_M, TILE_N, TILE_K)                        \
  template size_t genericFp8GemmKernelLauncherSm120<RET_TYPE, TILE_M, TILE_N, TILE_K>(            \
      __nv_fp8_e4m3 const* A, __nv_fp8_e4m3 const* B, float const* scale_a, float const* scale_b, \
      RET_TYPE* D, int m, int n, int k, int b, char* workspacePtr, size_t const workspaceBytes,   \
      cudaStream_t stream);

#endif  // FLASHINFER_FP8_GEMM_TEMPLATE_SM120_H_
