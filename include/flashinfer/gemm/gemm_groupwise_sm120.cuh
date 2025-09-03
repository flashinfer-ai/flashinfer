/*
 * Copyright (c) 2025 by FlashInfer team.
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
#ifndef FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_
#define FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_

#include <flashinfer/allocator.h>

#include <cassert>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <typeinfo>

#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace flashinfer {
namespace gemm {

// SM120 implementation with fixed scale granularity following CUTLASS examples
// For SM120, only ScaleGranularityM = 1, ScaleGranularityN = 128, ScaleGranularityK = 128 are
// supported
template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int MmaSM, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM120(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                            float* SFB_ptr, DTypeOut* D_ptr, int m, int n, int k,
                                            int l, cudaStream_t stream) {
  // SM120 only supports these specific scale granularities (per CUTLASS examples)
  static_assert(ScaleGranularityM == 1, "SM120 only supports ScaleGranularityM = 1");
  static_assert(ScaleGranularityN == 128 || ScaleGranularityN == 64 || ScaleGranularityN == 32 ||
                    ScaleGranularityN == 16,
                "SM120 only supports ScaleGranularityN = 128, 64, 32 or 16");
  static_assert(ScaleGranularityK == 128, "SM120 only supports ScaleGranularityK = 128");
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  using namespace cute;

  using ElementA = DTypeIn;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = DTypeIn;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = DTypeOut;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  using ElementAccumulator = float;
  using ElementCompute = float;

  // Use different tile shapes based on MmaSM like CUTLASS examples
  // MmaSM=1 uses 128x128x128, MmaSM=2 uses smaller tile to avoid register spilling
  using MmaTileShape_MNK =
      std::conditional_t<MmaSM == 1, Shape<_128, _128, _128>, Shape<_64, _128, _128>>;
  using ClusterShape_MNK = Shape<_1, _1, _1>;

  // Define blockwise scale configuration following CUTLASS pattern
  // SM120 needs explicit ScaleMajorK handling like SM100
  using ScaleConfig =
      std::conditional_t<ScaleMajorK,
                         cutlass::detail::Sm120BlockwiseScaleConfig<
                             ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
                             cute::UMMA::Major::K, cute::UMMA::Major::K>,
                         cutlass::detail::Sm120BlockwiseScaleConfig<
                             ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
                             cute::UMMA::Major::MN, cute::UMMA::Major::MN>>;

  // Use decltype like SM100 does for consistency
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC, AlignmentC, ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  // SM120 blockwise scaling uses auto stage count with epilogue carveout
  // This matches the CUTLASS example configuration
  using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK, StageCount,
      cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                           CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, l));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, l));

  // For beta=0 case, C and D can be the same buffer
  // This avoids allocating extra memory from workspace
  DTypeOut* C_ptr = D_ptr;  // Use D as both input and output when beta=0

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {A_ptr, stride_A, B_ptr, stride_B, SFA_ptr, layout_SFA, SFB_ptr, layout_SFB},
      {{}, C_ptr, stride_C, D_ptr, stride_D}};  // C and D point to same buffer when beta=0

  // Set alpha and beta for the epilogue
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;

  // Check device compute capability first
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);

  Gemm gemm;

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  size_t workspace_size = Gemm::get_workspace_size(arguments);

  if (workspace_size > float_buffer_size_in_bytes) {
    return cudaErrorInsufficientDriver;
  }

  // Pass workspace pointer only if needed
  void* kernel_workspace = nullptr;
  if (workspace_size > 0) {  // Only provide a pointer if workspace is actually needed
    kernel_workspace = float_buffer;
  }

  status = gemm.initialize(arguments, kernel_workspace);
  if (status != cutlass::Status::kSuccess) {
    // Don't continue if initialization failed
    return cudaErrorNotSupported;
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Sync to ensure kernel completes
  cudaError_t cuda_err = cudaStreamSynchronize(stream);
  if (cuda_err != cudaSuccess) {
    return cuda_err;
  }

  return cudaSuccess;
#else
  return cudaErrorNotSupported;
#endif
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GEMM_GROUPWISE_SM120_CUH_
