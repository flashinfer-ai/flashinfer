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
#ifndef FLASHINFER_GEMM_GROUPWISE_SM100_CUH_
#define FLASHINFER_GEMM_GROUPWISE_SM100_CUH_

#include <type_traits>
#include <utility>

#include "../allocator.h"
#include "../cutlass_utils.cuh"
#include "../utils.cuh"

namespace flashinfer {

namespace gemm {

using namespace cute;

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int MmaSM, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM100(void* float_buffer, size_t float_buffer_size_in_bytes,
                                            DTypeIn* A_ptr, DTypeIn* B_ptr, float* SFA_ptr,
                                            float* SFB_ptr, DTypeOut* D_ptr, int m, int n, int k,
                                            int l, cudaStream_t stream) {
  using ElementA = DTypeIn;                   // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = DTypeIn;                      // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementC = DTypeOut;                  // Element type for C and D matrix operands
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  // MMA type
  using ElementAccumulator = float;  // Element Accumulator will also be our scale factor type
  using ElementCompute = float;

  using MmaTileShape_MNK = Shape<cute::Int<MmaSM * 128>, _128, _128>;
  using ClusterShape_MNK = Shape<cute::Int<MmaSM>, _1, _1>;

  // NOTE(Zihao):: UMMA::Major::MN, UMMA::Major::MN is the fastest configuration.

  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
      cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK, UMMA::Major::MN,
                                                 UMMA::Major::MN>>;

  using LayoutSFA =
      decltype(ScaleConfig::deduce_layoutSFA());  // Layout type for SFA matrix operand
  using LayoutSFB =
      decltype(ScaleConfig::deduce_layoutSFB());  // Layout type for SFB matrix operand
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC, AlignmentC, ElementD, LayoutC, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB, cute::tuple<LayoutB, LayoutSFB>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelScheduleSm100Blockwise>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      void>;  // Default to ClusterLaunchControl (CLC) based tile scheduler

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                                     {m, n, k, l},
                                     {
                                         A_ptr,
                                         stride_A,
                                         B_ptr,
                                         stride_B,
                                         SFA_ptr,
                                         layout_SFA,
                                         SFB_ptr,
                                         layout_SFB,
                                     },
                                     {
                                         {},  // epilogue.thread
                                         D_ptr,
                                         stride_C,
                                         D_ptr,
                                         stride_C,
                                     }};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
  auto workspace_ptr = float_allocator.aligned_alloc<void>(workspace_size, 16,
                                                           "sm100_groupwise_gemm_float_workspace");

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));
  CUTLASS_CHECK(gemm.run(stream));
  return cudaSuccess;
}

template <int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK, bool ScaleMajorK,
          int MmaSM, typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGEMMSM100LowLatency(void* float_buffer,
                                                      size_t float_buffer_size_in_bytes,
                                                      DTypeIn* A_ptr, DTypeIn* B_ptr,
                                                      float* SFA_ptr, float* SFB_ptr,
                                                      DTypeOut* D_ptr, int m, int n, int k, int l,
                                                      cudaStream_t stream) {
  /*
    For small batch sizes (M) like 8, 16, 32 - typically we can only have at minimum M tile size of
    64 - because of tcgen05.mma shapes. This causes wasted work in the M dimension and less CTAs
    being able to do work. So one trick we can do is instead of calculating D = A @ B, we can
    calculate D.T = B.T @ A.T This allows us to use a smaller N tile size and more CTAs being able
    to do work. We can transpose by doing swapping row major and column major layouts. A: (m, k) row
    major    => (k, m) column major B: (k, n) column major => (n, k) row major D: (m, n) row major
    => (n, m) column major

    So instead of a row-column-row gemm we perform a row-column-column gemm.
  */

  // Compute the transpose GEMM:
  //   D^T = B^T @ A^T
  // so we swap (m, n) and swap (A, B) (+their scale tensors). The output is written as
  // column-major (n, m), which matches the original row-major (m, n) memory layout.
  std::swap(m, n);
  std::swap(A_ptr, B_ptr);
  std::swap(SFA_ptr, SFB_ptr);
  // Do the swap here as well
  using ScaleConfig = std::conditional_t<
      ScaleMajorK,
      cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityN, ScaleGranularityM,
                                                 ScaleGranularityK, UMMA::Major::K, UMMA::Major::K>,
      cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityN, ScaleGranularityM,
                                                 ScaleGranularityK, UMMA::Major::MN,
                                                 UMMA::Major::MN>>;

  using ElementA = DTypeIn;
  using LayoutA = cutlass::layout::RowMajor;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = DTypeIn;
  using LayoutB = cutlass::layout::ColumnMajor;
  constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = DTypeOut;
  using LayoutC = cutlass::layout::ColumnMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  // MMA type
  using ElementAccumulator = float;  // Element Accumulator will also be our scale factor type
  using ElementCompute = float;

  using MmaTileShape_MNK = Shape<cute::Int<128>, _16, _128>;
  using ClusterShape_MNK = Shape<int, int, _1>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD, cutlass::epilogue::TmaWarpSpecialized1Sm,
      cutlass::epilogue::fusion::LinearCombination<ElementD, float, void, float>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      /*ArchTag=*/cutlass::arch::Sm100, /*OpClass=*/cutlass::arch::OpClassTensorOp, ElementA,
      /*GemmLayoutA=*/cute::tuple<LayoutA, LayoutSFA>, AlignmentA, ElementB,
      /*GemmLayoutB=*/cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator,
      /*TileShapeMNK=*/MmaTileShape_MNK, ClusterShape_MNK,
      /*StageCountType=*/
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      /*KernelScheduleType=*/cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>::
      CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      /*ProblemShapeOrThreadblockMma_=*/Shape<int, int, int, int>, CollectiveMainloop,
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

  auto layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, l));
  auto layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, l));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {
          A_ptr,
          stride_A,
          B_ptr,
          stride_B,
          SFA_ptr,
          layout_SFA,
          SFB_ptr,
          layout_SFB,
      },
      {
          {},  // epilogue.thread
          nullptr,
          stride_C,
          D_ptr,
          stride_D,
      },
      // KernelHardwareInfo
      []() {
        // For some reason can_implement fails if this is not defined
        auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>();
        hw_info.cluster_shape = {1, 1, 1};
        hw_info.cluster_shape_fallback = {1, 1, 1};
        return hw_info;
      }(),
  };
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
  auto workspace_ptr = float_allocator.aligned_alloc<void>(
      workspace_size, 16, "sm100_groupwise_gemm_small_batch_size_float_workspace");

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));
  CUTLASS_CHECK(gemm.run(stream));
  return cudaSuccess;
}

}  // namespace gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_GROUPWISE_SM100_CUH_
